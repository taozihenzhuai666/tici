// 美白：只按颜色找出皮肤，把皮肤提亮、去黄；不认人、不改脸形、不动背景。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(1..3); Whiten.stop()
(function () {
  const LEVELS = { 1: .2, 2: .35, 3: .5 };
  let gl = null, cv = null, video = null, prog = null, uK = null, tex = null;
  let running = false, level = 2, vfcId = 0, rafId = 0;

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  const FS = `precision mediump float;
varying vec2 uv; uniform sampler2D vid; uniform float k;
void main() {
  vec3 c = texture2D(vid, uv).rgb;
  float y = dot(c, vec3(.299, .587, .114));
  // warm, mid-bright pixels count as skin; a wide soft ramp so the face lightens evenly, no patches
  float skin = smoothstep(.0, .18, c.r - c.b) * smoothstep(.12, .3, y) * (1. - smoothstep(.85, .98, y));
  vec3 w = c + (1. - c) * k * .5;                 // lighter
  w = mix(w, vec3(dot(w, vec3(.299, .587, .114))), k * .22);   // a little less yellow-orange, hue kept
  w.r += k * .015;                                // keep a healthy warmth
  c = mix(c, w, skin);
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
    uK = gl.getUniformLocation(prog, 'k');
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
      gl.uniform1f(uK, LEVELS[level] || LEVELS[2]);
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
      running = true; frame();
    },
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); vfcId = rafId = 0;
    },
  };
})();
