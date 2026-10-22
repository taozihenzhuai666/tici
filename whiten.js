// 美白：只给脸变白，别的地方一点不动，清晰度、对比度不降。
// 找脸不用模型：每隔几帧在一张很小的画面上按肤色找出最像脸的一块，只在这块椭圆里、而且是皮肤颜色的地方提亮。
// 提亮用的是中间调曲线：暗部、轮廓不变，所以不会发灰。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(1..3); Whiten.stop()
(function () {
  const LEVELS = { 1: .2, 2: .35, 3: .5 };
  let gl = null, cv = null, video = null, prog = null, uK = null, tex = null;
  let running = false, level = 2, vfcId = 0, rafId = 0;
  let uFace = null, face = null, frameN = 0;
  const gCv = document.createElement('canvas'), gCtx = gCv.getContext('2d', { willReadFrequently: true });

  // the biggest skin-coloured blob that isn't a whole wall: that's the face
  function findFace() {
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
      if (area < n * .01 || area > n * .45 || edges >= 3) continue;      // too small, or a wall
      const fill = area / ((x1 - x0 + 1) * (y1 - y0 + 1));
      if (fill < .35) continue;                                            // faces are solid, not stringy
      const cx = (x0 + x1 + 1) / 2 / gw, cy = (y0 + y1 + 1) / 2 / gh;
      const score = area * (1.3 - Math.abs(cx - .5)) * (1.3 - Math.abs(cy - .45));   // prefer the middle
      if (score > bestScore) { bestScore = score; best = [cx, cy, (x1 - x0 + 1) / 2 / gw, (y1 - y0 + 1) / 2 / gh]; }
    }
    if (best) face = face ? face.map((v, i) => v + (best[i] - v) * .35) : best;
    else if (face) { face[2] *= .85; face[3] *= .85; if (face[2] < .02) face = null; }
  }

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  const FS = `precision highp float;
varying vec2 uv; uniform sampler2D vid; uniform float k;
uniform vec4 face;   // centre x, y (origin top-left), half width, half height; z = 0 means no face found
const vec3 LUM = vec3(.299, .587, .114);
void main() {
  vec3 c = texture2D(vid, uv).rgb;
  if (k > 0. && face.z > 0.) {
    vec2 d = (uv - face.xy) / vec2(face.z * 1.15, face.w * 1.15);
    float inFace = 1. - smoothstep(.8, 1.1, length(d));
    float skin = smoothstep(.0, .15, c.r - c.b) * smoothstep(.06, .22, dot(c, LUM));
    float a = inFace * skin;
    if (a > 0.) {
      vec3 w = pow(max(c, vec3(0.)), vec3(1. - k * .55));   // brighter mid-tones; black stays black
      w = mix(w, vec3(dot(w, LUM)), k * .18);               // a little less yellow
      w = mix(w, w * w * (3. - 2. * w), .25);               // keep the face's own contrast
      c = mix(c, w, a);
    }
  }
  gl_FragColor = vec4(c, 1.);
}`;

  function init(canvas) {
    if (gl && cv === canvas) return;
    cv = canvas;
    gl = canvas.getContext('webgl', { antialias: false, alpha: false, premultipliedAlpha: false });
    if (!gl) throw new Error('这个浏览器不支持 WebGL');
    const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
      if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog); gl.useProgram(prog);
    uK = gl.getUniformLocation(prog, 'k'); uFace = gl.getUniformLocation(prog, 'face');
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
      if (frameN++ % 4 === 0) findFace();
      gl.uniform1f(uK, LEVELS[level] || LEVELS[2]);
      const f = face || [0, 0, 0, 0];
      gl.uniform4f(uFace, f[0], f[1], f[2], f[3]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    if (video.requestVideoFrameCallback) vfcId = video.requestVideoFrameCallback(frame);
    else rafId = requestAnimationFrame(frame);
  }

  window.Whiten = {
    isRunning: () => running,
    setLevel(n) { level = n; },
    start(v, canvas) {
      init(canvas); video = v;
      if (running) return;
      running = true; face = null; frame();
    },
    debug: () => ({ face }),
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); vfcId = rafId = 0;
    },
  };
})();
