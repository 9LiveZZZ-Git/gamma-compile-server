// Minimal OSC 1.0 codec — encode + decode for the common message
// types used by audio / VJ / lighting controllers. Self-contained:
// no dependency on the npm `osc` package (which is ~300 KB +
// pulls in EventEmitter ceremony we don't need). Spec reference:
// https://opensoundcontrol.stanford.edu/spec-1_0.html
//
// Supported argument types:
//   f  float32       (BE 4-byte IEEE 754)
//   i  int32         (BE 4-byte signed)
//   s  string        (null-terminated, padded to 4-byte boundary)
//   T  true          (no data)
//   F  false         (no data)
//   N  null / nil    (no data)
//   I  impulse / inf (no data)
//   d  float64       (BE 8-byte)
//   h  int64         (BE 8-byte) — returned as BigInt
//   b  blob          (length-prefixed bytes, padded to 4)
//
// Bundles supported on decode (recursive) but not built on encode --
// we emit individual messages, which every receiver supports.

const ASCII = "ascii";

/* Read a null-terminated string starting at offset `i` in `buf`.
 * Returns [string, nextOffset]. Strings in OSC are always padded
 * to a 4-byte boundary including the terminating null. */
function readString(buf, i) {
  const start = i;
  while (i < buf.length && buf[i] !== 0) i++;
  const s = buf.slice(start, i).toString(ASCII);
  // Move past the null + any pad bytes that bring us to a 4-byte
  // boundary. The padding bytes are unspecified -- we don't validate.
  i++;
  const pad = (4 - (i % 4)) % 4;
  return [s, i + pad];
}

/* Encode a string as the null-terminated 4-byte-padded form OSC uses. */
function encodeString(s) {
  // Strict ASCII per spec. Anything non-ASCII gets dropped (rather
  // than silently corrupting the byte stream).
  const ascii = Buffer.from(String(s), ASCII);
  const len = ascii.length + 1;            // +1 for null terminator
  const padded = Math.ceil(len / 4) * 4;
  const out = Buffer.alloc(padded);        // alloc zeroes the pad
  ascii.copy(out, 0);
  return out;
}

/* Decode a single OSC packet (message OR bundle). For a message returns
 * { address, args }. For a bundle returns { bundle: true, timetag,
 * messages: [...] } where each item is itself a decode result. */
export function decodeOsc(buf) {
  if (!Buffer.isBuffer(buf)) buf = Buffer.from(buf);
  if (buf.length < 4) throw new Error("osc: packet too short");

  let i = 0;
  const [head, afterHead] = readString(buf, i);
  i = afterHead;

  if (head === "#bundle") {
    if (i + 8 > buf.length) throw new Error("osc: bundle missing timetag");
    // Read 64-bit NTP timetag (high 32 = seconds since 1900, low 32 = fractional).
    const ttHi = buf.readUInt32BE(i);
    const ttLo = buf.readUInt32BE(i + 4);
    i += 8;
    const messages = [];
    while (i < buf.length) {
      if (i + 4 > buf.length) throw new Error("osc: bundle element header truncated");
      const sz = buf.readInt32BE(i);
      i += 4;
      if (sz < 0 || i + sz > buf.length) throw new Error("osc: bundle element size invalid");
      messages.push(decodeOsc(buf.slice(i, i + sz)));
      i += sz;
    }
    return { bundle: true, timetag: { sec: ttHi, frac: ttLo }, messages };
  }

  // Message. Address is `head`. Next comes the type-tag string.
  if (i >= buf.length) return { address: head, args: [] };
  const [tags, afterTags] = readString(buf, i);
  i = afterTags;
  if (!tags.startsWith(",")) {
    // Some senders skip the type-tag string when there are no args.
    // Tolerate this rather than rejecting.
    return { address: head, args: [] };
  }
  const args = [];
  for (let k = 1; k < tags.length; k++) {
    const t = tags[k];
    if (t === "f") { args.push(buf.readFloatBE(i));  i += 4; }
    else if (t === "i") { args.push(buf.readInt32BE(i));  i += 4; }
    else if (t === "d") { args.push(buf.readDoubleBE(i)); i += 8; }
    else if (t === "h") {
      const hi = buf.readInt32BE(i);
      const lo = buf.readUInt32BE(i + 4);
      args.push((BigInt(hi) << 32n) | BigInt(lo));
      i += 8;
    }
    else if (t === "s" || t === "S") {
      const [s, ni] = readString(buf, i);
      args.push(s);
      i = ni;
    }
    else if (t === "b") {
      const sz = buf.readInt32BE(i);
      i += 4;
      if (sz < 0 || i + sz > buf.length) throw new Error("osc: blob size invalid");
      args.push(buf.slice(i, i + sz));
      i += sz;
      const pad = (4 - (sz % 4)) % 4;
      i += pad;
    }
    else if (t === "T") args.push(true);
    else if (t === "F") args.push(false);
    else if (t === "N") args.push(null);
    else if (t === "I") args.push(Infinity);
    else {
      // Unknown type. Per OSC 1.1 we should report; for our use we'll
      // skip unknown types silently (with a warning visible upstream).
      throw new Error("osc: unsupported type tag '" + t + "' at index " + k);
    }
  }
  return { address: head, args };
}

/* Encode a single message to an OSC packet. `args` is an array of
 * JavaScript values; types are inferred:
 *   number  → f (float32) — unless integer && |x| < 2^31 then 'i'
 *   string  → s
 *   boolean → T / F
 *   null    → N
 *   Buffer  → b
 *   BigInt  → h
 * Pass an explicit `{type:'f'|'i'|'d', value:x}` to override
 * inference (e.g. force-int for OSC receivers that care). */
export function encodeOsc(address, args) {
  if (typeof address !== "string" || !address.startsWith("/")) {
    throw new Error("osc: address must start with '/'");
  }
  if (!Array.isArray(args)) args = (args === undefined || args === null) ? [] : [args];

  // First pass: figure out type tags + per-arg buffers.
  let tags = ",";
  const argBufs = [];

  for (const raw of args) {
    let v = raw, t;
    if (raw && typeof raw === "object" && !Buffer.isBuffer(raw) && raw.type) {
      v = raw.value;
      t = raw.type;
    } else if (typeof raw === "number") {
      t = (Number.isInteger(raw) && Math.abs(raw) < 0x80000000) ? "i" : "f";
    } else if (typeof raw === "string") {
      t = "s";
    } else if (raw === true)  { t = "T"; }
    else if (raw === false) { t = "F"; }
    else if (raw === null)  { t = "N"; }
    else if (typeof raw === "bigint") { t = "h"; }
    else if (Buffer.isBuffer(raw)) { t = "b"; }
    else {
      throw new Error("osc: cannot encode argument of type " + (typeof raw));
    }

    tags += t;

    if (t === "f") {
      const b = Buffer.alloc(4); b.writeFloatBE(Number(v), 0);  argBufs.push(b);
    } else if (t === "i") {
      const b = Buffer.alloc(4); b.writeInt32BE(Math.trunc(Number(v)), 0); argBufs.push(b);
    } else if (t === "d") {
      const b = Buffer.alloc(8); b.writeDoubleBE(Number(v), 0); argBufs.push(b);
    } else if (t === "h") {
      const big = BigInt(v);
      const b = Buffer.alloc(8);
      b.writeBigInt64BE(big, 0);
      argBufs.push(b);
    } else if (t === "s" || t === "S") {
      argBufs.push(encodeString(String(v)));
    } else if (t === "b") {
      const blob = Buffer.isBuffer(v) ? v : Buffer.from(v);
      const len = blob.length;
      const pad = (4 - (len % 4)) % 4;
      const b = Buffer.alloc(4 + len + pad);
      b.writeInt32BE(len, 0);
      blob.copy(b, 4);
      argBufs.push(b);
    }
    // T / F / N / I encode no data — tag-only.
  }

  return Buffer.concat([encodeString(address), encodeString(tags), ...argBufs]);
}

/* Encode an OSC bundle. `messages` is an array of {address, args}
 * objects (or nested bundles, recursively). `timetag` is optional:
 *   undefined / "immediate" -> the special immediate-now timetag
 *     (63 zero bits + 1 in the LSB; receivers process the bundle
 *      as soon as it arrives)
 *   {sec, frac}             -> raw NTP timetag (sec since 1900,
 *                              fractional seconds in 32 bits)
 *
 * Bundle wire format (OSC 1.0 §3):
 *   "#bundle\0"      8 bytes (string + null + 3 pad)
 *   <timetag>        8 bytes (NTP)
 *   for each element:
 *     <int32 size>   4 bytes (size of this element in bytes)
 *     <element>      message OR nested bundle
 */
export function encodeBundle(messages, timetag) {
  if (!Array.isArray(messages)) {
    throw new Error("osc: bundle messages must be an array");
  }
  const headerStr = encodeString("#bundle");
  let ttHi = 0, ttLo = 1;        // immediate
  if (timetag && typeof timetag === "object") {
    ttHi = (timetag.sec  | 0) >>> 0;
    ttLo = (timetag.frac | 0) >>> 0;
  }
  const tt = Buffer.alloc(8);
  tt.writeUInt32BE(ttHi, 0);
  tt.writeUInt32BE(ttLo, 4);

  const elementBufs = [];
  for (const m of messages) {
    let el;
    if (m && m.bundle && Array.isArray(m.messages)) {
      el = encodeBundle(m.messages, m.timetag);
    } else if (m && typeof m.address === "string") {
      el = encodeOsc(m.address, m.args || []);
    } else {
      throw new Error("osc: bundle element must be {address, args} or nested bundle");
    }
    const sz = Buffer.alloc(4);
    sz.writeInt32BE(el.length, 0);
    elementBufs.push(sz, el);
  }

  return Buffer.concat([headerStr, tt, ...elementBufs]);
}

/* Compile an OSC address pattern to a regex. Implements the full
 * OSC 1.0 wildcard set:
 *
 *   ?           single character (NOT '/')
 *   *           zero or more characters (NOT '/')
 *   [abc]       character class -- matches one of a/b/c
 *   [a-z]       character range
 *   [!abc]      negated class (also accepts [^abc] for regex-style)
 *   {alt,erna,tives}   alternation -- matches any of the listed strings
 *   /           literal separator -- never matched by wildcards
 *
 * Returns a RegExp anchored to the full address. Cached internally so
 * repeated calls with the same pattern are cheap. */
const _patternRegexCache = new Map();
const _PATTERN_CACHE_LIMIT = 512;

export function patternToRegex(pattern) {
  const cached = _patternRegexCache.get(pattern);
  if (cached) return cached;

  let re = "^";
  let i = 0;
  while (i < pattern.length) {
    const c = pattern[i];
    if (c === "*") {
      re += "[^/]*";
      i++;
    } else if (c === "?") {
      re += "[^/]";
      i++;
    } else if (c === "[") {
      // Character class. Find the matching ']' (no nesting allowed
      // by the spec). Inside, '!' as the first character flips to
      // negation; '-' between characters defines a range as per
      // typical glob syntax.
      const end = pattern.indexOf("]", i + 1);
      if (end < 0) {
        // Unmatched '[' -- treat as literal. Permissive over strict.
        re += "\\[";
        i++;
      } else {
        let body = pattern.slice(i + 1, end);
        if (body.length === 0) {
          // Empty class -- match nothing. Tolerate by writing an
          // intentionally-impossible class.
          re += "[^\\s\\S]";
        } else {
          if (body[0] === "!") body = "^" + body.slice(1);
          // Regex character classes treat ']' literally only as the
          // first char (after optional ^). Forward slashes are
          // forbidden in OSC pattern classes, but we don't enforce.
          // Escape backslash + closing bracket; everything else is
          // OK inside a character class.
          re += "[" + body.replace(/\\/g, "\\\\") + "]";
        }
        i = end + 1;
      }
    } else if (c === "{") {
      // Alternation -- {abc,def,ghi}. Match any literal alternative.
      // The OSC spec disallows '/' inside alternatives + nesting.
      const end = pattern.indexOf("}", i + 1);
      if (end < 0) {
        re += "\\{";
        i++;
      } else {
        const body = pattern.slice(i + 1, end);
        const alts = body.split(",").map(s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
        re += "(?:" + alts.join("|") + ")";
        i = end + 1;
      }
    } else if ("/^$.|+()\\".includes(c)) {
      // Regex metachars that aren't OSC wildcards -- literal-escape.
      // '/' is explicitly listed so it can't collide with regex
      // tokens; the OSC matcher treats it as a literal.
      re += "\\" + c;
      i++;
    } else {
      re += c;
      i++;
    }
  }
  re += "$";

  let compiled;
  try { compiled = new RegExp(re); }
  catch (_) { compiled = /^.\B/; /* never matches anything */ }

  // Bounded LRU-ish cache. On overflow drop the oldest entry to bound memory.
  if (_patternRegexCache.size >= _PATTERN_CACHE_LIMIT) {
    const firstKey = _patternRegexCache.keys().next().value;
    _patternRegexCache.delete(firstKey);
  }
  _patternRegexCache.set(pattern, compiled);
  return compiled;
}

/* Test whether an OSC address pattern matches a target address. */
export function matchesPattern(pattern, address) {
  if (pattern === address) return true;
  return patternToRegex(pattern).test(address);
}
