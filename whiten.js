// 美白 + 瘦脸，只动脸（和脖子上露出的皮肤），别的地方（墙、衣服、背景）一点不动，清晰度、对比度不降。
// 找脸：用 MediaPipe 人脸检测（文件在本站 mp/，开美白或瘦脸时加载，用 CPU——iPhone 上 GPU 会卡住不动）。
//       模型能用时只信模型：没找到脸就什么都不改，不会去动墙、抱枕。
//       模型实在加载不了，才用保底办法按肤色找脸，而且只做美白，不做瘦脸（找错了扭曲很难看）。
// 脸朝哪边：不靠猜。模型给出眼睛和嘴的位置，从眼睛指向嘴就是下巴方向——手机怎么拿、画面怎么转都对得上，
//       脸是倒着、歪着都能认出来，不会再锁在错的方向上。
// 美白：脸和脖子范围里、而且是皮肤颜色的地方提亮、去黄（暗部、轮廓不变，不发灰）。
// 瘦脸：两颊、下巴往里收，按真正的下巴方向收。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(0..3); Whiten.setSlim(true); Whiten.setTurn(度数); Whiten.loadFace(); Whiten.stop()
(function () {
  const LEVELS = { 1: .3, 2: .5, 3: .7 };
  const SLIM = .09;     // cheeks and jaw move in about 9%
  const LOAD_LIMIT = 30000;
  const here = p => new URL(p, document.baseURI).href;
  let gl = null, cv = null, video = null, prog = null, tex = null, U = {};
  let running = false, level = 2, slimOn = false, turnDeg = 0, vfcId = 0, rafId = 0;
  let fd = null, faceLoading = null, modelState = 'idle', loadErr = '', loadMs = 0, tries = 0;
  // face, in the camera picture's own pixels: centre x, y; chin direction dx, dy (unit); half width, half length
  let face = null, faceBy = '', frameN = 0, lastTs = 0, found = 0, looked = 0;
  const gCv = document.createElement('canvas'), gCtx = gCv.getContext('2d', { willReadFrequently: true });

  function loadFace() {
    if (fd) return Promise.resolve();
    if (faceLoading) return faceLoading;
    modelState = 'loading'; loadErr = ''; tries++;
    const t0 = performance.now();
    const load = (async () => {
      const mp = await import(here('mp/vision_bundle.mjs'));
      const files = await mp.FilesetResolver.forVisionTasks(here('mp/wasm'));
      const d = await mp.FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: here('mp/blaze_face_short_range.tflite'), delegate: 'CPU' },
        runningMode: 'VIDEO', minDetectionConfidence: .5,
      });
      fd = d; modelState = 'ready'; loadMs = Math.round(performance.now() - t0); face = null;
    })();
    // a slow phone or network: say so after 30 s, but keep the load going and use it if it still arrives
    const limit = new Promise((_, no) => setTimeout(() => no(new Error('加载超过 30 秒')), LOAD_LIMIT));
    load.catch(e => { loadErr = (e && e.message) || String(e); });
    faceLoading = Promise.race([load, limit]).catch(e => {
      if (!fd) { modelState = 'failed'; loadErr = loadErr || (e && e.message) || String(e); }
      faceLoading = null;
      if (fd) return;
      // try again by itself a couple of times (a dropped download, a busy phone)
      if (tries < 3) setTimeout(() => { if (!fd && !faceLoading) loadFace().catch(() => {}); }, 5000);
      throw e;
    });
    return faceLoading;
  }

  // follow the face smoothly; jump when it's a new one
  function glide(nf, by) {
    if (!face || faceBy !== by || Math.hypot(nf[0] - face[0], nf[1] - face[1]) > nf[4] * 1.5) face = nf.slice();
    else {
      for (let i = 0; i < 6; i++) face[i] += (nf[i] - face[i]) * .45;
      const n = Math.hypot(face[2], face[3]) || 1; face[2] /= n; face[3] /= n;
    }
    faceBy = by;
  }
  function fade() { if (face) { face[4] *= .85; face[5] *= .85; if (face[4] < 6) face = null; } }

  // The model knows upright faces best, so the picture is shrunk (quicker) and turned to stand the person
  // up first. Which turn: the model will also "find" an upside-down face, with the eyes and mouth in the
  // wrong places, so when a face first shows up all four turns are compared and the surest one is kept;
  // after that the kept turn is looked at, and every ~3 s it's checked against upside down.
  const rCv = document.createElement('canvas'), rCtx = rCv.getContext('2d');
  let tryT = 0, misses = 0, locked = false, sinceScan = 0;
  function detectAt(t) {
    const W = video.videoWidth, H = video.videoHeight;
    const s = Math.min(1, 480 / Math.max(W, H)), w = Math.round(W * s), h = Math.round(H * s);
    const side = t % 180 !== 0, sw = side ? h : w, sh = side ? w : h;
    if (rCv.width !== sw || rCv.height !== sh) { rCv.width = sw; rCv.height = sh; }
    rCtx.setTransform(1, 0, 0, 1, sw / 2, sh / 2); rCtx.rotate(t * Math.PI / 180);
    rCtx.drawImage(video, -w / 2, -h / 2, w, h);
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    looked++;
    let best = null;
    for (const x of fd.detectForVideo(rCv, lastTs).detections || [])
      if (x.keypoints && x.keypoints.length >= 4 && (!best || x.boundingBox.width > best.boundingBox.width)) best = x;
    if (!best) return null;
    const score = (best.categories && best.categories[0] && best.categories[0].score) || 0;
    return { t, s, sw, sh, W, H, det: best, score };
  }
  const ORDER = [0, 180, 90, 270];
  let scanI = 0;
  // all four ways (or just the kept way and upside down), the surest wins
  function surest(from, turns) {
    let hit = null;
    for (const r of turns) { const h = detectAt((from + r) % 360); if (h && (!hit || h.score > hit.score + .02)) hit = h; }
    return hit;
  }
  function findFaceModel() {
    let hit;
    try {
      if (!locked) {
        // nobody yet: one way per look, in turn (cheap); once something shows up, compare all four
        hit = detectAt((turnDeg + ORDER[scanI++ % 4]) % 360);
        if (hit) hit = surest(turnDeg, ORDER);
      } else if (++sinceScan >= 60) { sinceScan = 0; hit = surest(tryT, [0, 180]); }
      else hit = detectAt(tryT);
    } catch (e) { loadErr = '识别出错：' + e.message; return fade(); }
    if (!hit) {
      fade();
      if (locked && ++misses >= 40) { misses = 0; locked = false; scanI = 0; }   // gone ~2 s: look all ways again
      return;
    }
    misses = 0; locked = true; found++; tryT = hit.t;
    const { t, s, sw, sh, W, H, det } = hit;
    // turned-picture pixels -> camera-picture pixels
    const a = -t * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    const back = (px, py) => {
      const x = px - sw / 2, y = py - sh / 2;
      return [W / 2 + (x * ca - y * sa) / s, H / 2 + (x * sa + y * ca) / s];
    };
    const k = det.keypoints, bb = det.boundingBox;
    const P = i => back(k[i].x * sw, k[i].y * sh);
    const eR = P(0), eL = P(1), mo = P(3);
    let dx = mo[0] - (eR[0] + eL[0]) / 2, dy = mo[1] - (eR[1] + eL[1]) / 2;
    const n = Math.hypot(dx, dy) || 1; dx /= n; dy /= n;
    // the chin points down in the turned picture; eyes and mouth only add the head's tilt (keep it within 40°)
    const dn = [Math.sin(t * Math.PI / 180), Math.cos(t * Math.PI / 180)];
    if (dx * dn[0] + dy * dn[1] < .77) { dx = dn[0]; dy = dn[1]; }
    const hw = bb.width / 2 / s, hb = bb.height / 2 / s;
    const c = back(bb.originX + bb.width / 2, bb.originY + bb.height / 2);
    // the model's box stops at the eyebrows: stretch it up over the forehead
    glide([c[0] - dx * hb * .25, c[1] - dy * hb * .25, dx, dy, hw, hb * 1.25], 'model');
    lastScore = Math.round(hit.score * 100) / 100;
  }
  let lastScore = 0;

  // without the model: the biggest solid skin-coloured blob that isn't a whole wall
  function findFaceSkin() {
    const W = video.videoWidth, H = video.videoHeight;
    const gw = 48, gh = Math.max(16, Math.round(48 * H / W)), n = gw * gh;
    if (gCv.width !== gw || gCv.height !== gh) { gCv.width = gw; gCv.height = gh; }
    gCtx.drawImage(video, 0, 0, gw, gh);
    const px = gCtx.getImageData(0, 0, gw, gh).data;
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2];
      const y = .299 * r + .587 * g + .114 * b;
      const cb = 128 - .168736 * r - .331264 * g + .5 * b, cr = 128 + .5 * r - .418688 * g - .081312 * b;
      m[i] = cr >= 136 && cr <= 175 && cb >= 77 && cb <= 127 && y > 45 && y < 240 && r > g && r - b > 12 ? 1 : 0;
    }
    const seen = new Uint8Array(n), stack = [];
    let best = null, bestScore = 0;
    for (let s0 = 0; s0 < n; s0++) {
      if (!m[s0] || seen[s0]) continue;
      let area = 0, x0 = gw, x1 = 0, y0 = gh, y1 = 0;
      stack.push(s0); seen[s0] = 1;
      while (stack.length) {
        const i = stack.pop(), x = i % gw, y = (i / gw) | 0;
        area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0 && m[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; stack.push(i - 1); }
        if (x < gw - 1 && m[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; stack.push(i + 1); }
        if (y > 0 && m[i - gw] && !seen[i - gw]) { seen[i - gw] = 1; stack.push(i - gw); }
        if (y < gh - 1 && m[i + gw] && !seen[i + gw]) { seen[i + gw] = 1; stack.push(i + gw); }
      }
      const edges = (x0 === 0) + (x1 === gw - 1) + (y0 === 0) + (y1 === gh - 1);
      if (area < n * .01 || area > n * .45 || edges >= 3) continue;        // too small, or a wall
      if (area / ((x1 - x0 + 1) * (y1 - y0 + 1)) < .35) continue;            // faces are solid, not stringy
      const cx = (x0 + x1 + 1) / 2 / gw, cy = (y0 + y1 + 1) / 2 / gh;
      const score = area * (1.3 - Math.abs(cx - .5)) * (1.3 - Math.abs(cy - .45));   // prefer the middle
      if (score > bestScore) { bestScore = score; best = [cx * W, cy * H, (x1 - x0 + 1) / 2 / gw * W, (y1 - y0 + 1) / 2 / gh * H]; }
    }
    // which way is down isn't known here; whitening doesn't need it, so use the expected turn
    const tr = turnDeg * Math.PI / 180, dx = Math.sin(tr), dy = Math.cos(tr);
    if (best) glide([best[0], best[1], dx, dy, Math.min(best[2], best[3]), Math.max(best[2], best[3])], 'skin'); else fade();
  }

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  // everything in the camera picture's pixels (origin top-left), so a face at any angle works
  const FS = `precision highp float;
varying vec2 uv; uniform sampler2D vid; uniform float k, slim;
uniform vec2 res;     // picture size
uniform vec4 face;    // centre x, y; half width, half length (pixels); z = 0 means no face
uniform vec2 chin;    // which way the chin points (unit)
const vec3 LUM = vec3(.299, .587, .114);
vec2 warp(vec2 p) {
  if (face.z <= 0. || slim <= 0.) return p;
  vec2 across = vec2(-chin.y, chin.x);                   // cheek to cheek
  vec2 r = p - (face.xy + chin * face.w * .3);           // cheeks and jaw
  float u = dot(r, across) / face.z, v = dot(r, chin) / face.w;
  float w = 1. - smoothstep(.5, 1.3, length(vec2(u / 1.3, v / 1.05)));
  w *= .55 + .45 * smoothstep(-.6, .4, v);              // more at the jaw than at the cheekbones
  return p + across * dot(r, across) * slim * w;         // sample further out = cheeks move in
}
void main() {
  vec2 p = warp(uv * res);
  vec2 q = p / res;
  vec3 c = texture2D(vid, q).rgb;
  if (k > 0. && face.z > 0.) {
    vec2 across = vec2(-chin.y, chin.x);
    vec2 r = p - face.xy;
    float u = dot(r, across) / face.z, v = dot(r, chin) / face.w;
    float inFace = 1. - smoothstep(.85, 1.2, length(vec2(u / 1.2, v / 1.15)));
    // the neck under the chin too, so the face isn't whiter than the neck
    float inNeck = (1. - smoothstep(.75, 1.1, abs(u) / .95)) * (1. - smoothstep(1.9, 2.5, v)) * step(0., v);
    float skin = smoothstep(.01, .12, c.r - c.b) * smoothstep(.06, .2, dot(c, LUM)) * smoothstep(-.02, .04, c.r - c.g);
    float a = max(inFace, inNeck) * skin;
    if (a > 0.) {
      vec3 w = pow(max(c, vec3(0.)), vec3(1. - k * .6));    // brighter mid-tones; black stays black
      w = mix(w, vec3(dot(w, LUM)), k * .25);               // less yellow
      w.b += k * .05 * (1. - w.b);                          // a touch cooler, like fair skin
      w = mix(w, w * w * (3. - 2. * w), .2);                // keep the face's own contrast
      c = mix(c, w, a);
    }
  }
  gl_FragColor = vec4(c, 1.);
}`;

  function init(canvas) {
    if (gl && cv === canvas) return;
    cv = canvas;
    gl = canvas.getContext('webgl', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true });   // kept after drawing, so the turned recording can copy it
    if (!gl) throw new Error('这个浏览器不支持 WebGL');
    const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
      if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog); gl.useProgram(prog);
    for (const n of ['k', 'slim', 'res', 'face', 'chin']) U[n] = gl.getUniformLocation(prog, n);
    gl.uniform1i(gl.getUniformLocation(prog, 'vid'), 0);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  function frame() {
    if (!running) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H && video.readyState >= 2) {
      if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
      gl.viewport(0, 0, W, H);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
      if (frameN++ % 3 === 0) {
        if (fd) findFaceModel();
        else if (modelState === 'failed' && level) findFaceSkin();   // last resort, whitening only
        else face = null;                                           // model still loading: leave the picture alone
      }
      const f = face || [0, 0, 0, 1, 0, 0];
      gl.uniform1f(U.k, LEVELS[level] || 0);
      gl.uniform1f(U.slim, slimOn && faceBy === 'model' ? SLIM : 0);
      gl.uniform2f(U.res, W, H);
      gl.uniform2f(U.chin, f[2], f[3]);
      gl.uniform4f(U.face, f[0], f[1], f[4], f[5]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    if (video.requestVideoFrameCallback) vfcId = video.requestVideoFrameCallback(frame);
    else rafId = requestAnimationFrame(frame);
  }

  window.Whiten = {
    isRunning: () => running,
    setLevel(n) { level = n; },
    setSlim(on) { slimOn = on; },
    setTurn(deg) {
      deg = ((deg % 360) + 360) % 360;
      if (deg !== turnDeg) { turnDeg = deg; tryT = deg; misses = 0; locked = false; scanI = 0; face = null; }
    },
    loadFace,
    // what's happening, for the buttons: model 'idle' | 'loading' | 'ready' | 'failed'; face found or not
    status: () => ({ model: modelState, face: !!face, by: faceBy }),
    // for the long-press details
    debug: () => ({ model: modelState, err: loadErr, loadMs, tries, looked, found, turn: turnDeg, tryT, score: lastScore, face: face && face.map(v => Math.round(v * 100) / 100), by: faceBy,
      size: video ? video.videoWidth + '×' + video.videoHeight : '' }),
    start(v, canvas) {
      init(canvas); video = v;
      if (running) return;
      running = true; face = null; frame();
    },
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); vfcId = rafId = 0;
    },
  };
})();
