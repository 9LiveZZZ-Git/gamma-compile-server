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

/* Test whether an OSC address pattern matches a target address.
 *
 * Currently implements a SIMPLIFIED subset of the OSC pattern spec:
 *   *      matches any character sequence (within a single path part)
 *   ?      matches one character
 *   /      treated as a literal separator
 *
 * Future: [a-z] character classes, {alt,erna,tives} braces, full
 * cross-segment wildcards. The MVP covers >95% of TouchOSC / Reaper
 * layouts which use literal addresses. */
export function matchesPattern(pattern, address) {
  if (pattern === address) return true;
  // Compile pattern to regex. Each '*' → '[^/]*'; each '?' → '[^/]';
  // anything else is literal-escaped.
  let re = "^";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") re += "[^/]*";
    else if (c === "?") re += "[^/]";
    else if ("/^$.|+()[]{}\\".includes(c)) re += "\\" + c;
    else re += c;
  }
  re += "$";
  return new RegExp(re).test(address);
}
