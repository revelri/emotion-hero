/**
 * WebGL2 shaders for the emotional visualization system.
 *
 * Contains vertex shader, 18 fragment shader variants (switchable demo modes),
 * and simulation shaders (reaction-diffusion, smoke, erosion, LBM).
 */

export const vertexShaderSource = `#version 300 es
in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

/** Shader mode names for UI display */
export const SHADER_MODE_NAMES = [
    'Voronoi',
    'Curl Noise',
    'Domain-Warped FBM',
    'Metaballs',
    'Flow Field',
    'Reaction-Diffusion',
    'Chladni',
    'Cymatics',
    'Julia Set',
    'Gravity Lens',
    'Attractor',
    'Cracks',
    'Smoke',
    'Topography',
    'Magnetic LIC',
    'Lissajous',
    'Phyllotaxis',
    'Ink Flow',
];

// ---------------------------------------------------------------------------
// Shared GLSL preamble: uniforms, noise, OKLab, motion-vector feedback
// ---------------------------------------------------------------------------

const shaderPreamble = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D u_asciiMask;
uniform vec3 u_emotionColors[5];
uniform float u_emotionValues[5];
uniform float u_time;
uniform float u_emotionVelocities[5];
uniform sampler2D uPrevFrame;
uniform float uFeedbackStrength;
uniform float uEnableFeedback;
out vec4 fragColor;

// --- Simplex noise ---
vec3 permute(vec3 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                      + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// --- Curl noise (divergence-free flow) ---
vec2 curlNoise(vec2 p, float time) {
    float eps = 0.01;
    float n1 = snoise(vec2(p.x, p.y + eps) + time * 0.2);
    float n2 = snoise(vec2(p.x, p.y - eps) + time * 0.2);
    float n3 = snoise(vec2(p.x + eps, p.y) + time * 0.2);
    float n4 = snoise(vec2(p.x - eps, p.y) + time * 0.2);
    float dx = (n1 - n2) / (2.0 * eps);
    float dy = (n3 - n4) / (2.0 * eps);
    return vec2(dx, -dy);
}

// --- FBM (Fractal Brownian Motion) ---
float fbm(vec2 p, float time) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++) {
        value += amplitude * snoise(p * frequency + time * 0.15);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// --- sRGB / linear conversion ---
float srgbToLinear(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}
float linearToSrgb(float c) {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}
vec3 srgbToLinearVec3(vec3 rgb) {
    return vec3(srgbToLinear(rgb.r), srgbToLinear(rgb.g), srgbToLinear(rgb.b));
}
vec3 linearToSrgbVec3(vec3 rgb) {
    return vec3(linearToSrgb(rgb.r), linearToSrgb(rgb.g), linearToSrgb(rgb.b));
}

// --- OKLab color space ---
vec3 rgbToOkLab(vec3 rgb) {
    vec3 lms;
    lms.r = 0.4122214708 * rgb.r + 0.5363325363 * rgb.g + 0.0514459929 * rgb.b;
    lms.g = 0.2119034982 * rgb.r + 0.6806995451 * rgb.g + 0.1073969566 * rgb.b;
    lms.b = 0.0883024619 * rgb.r + 0.2817188376 * rgb.g + 0.6299787005 * rgb.b;
    lms = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    vec3 lab;
    lab.r = 0.2104542553 * lms.r + 0.7936177850 * lms.g - 0.0040720468 * lms.b;
    lab.g = 1.9779984951 * lms.r - 2.4285922050 * lms.g + 0.4505937099 * lms.b;
    lab.b = 0.0259040371 * lms.r + 0.7827717662 * lms.g - 0.8086757660 * lms.b;
    return lab;
}
vec3 okLabToRgb(vec3 lab) {
    vec3 lms;
    lms.r = lab.r + 0.3963377774 * lab.g + 0.2158037573 * lab.b;
    lms.g = lab.r - 0.1055613458 * lab.g - 0.0638541728 * lab.b;
    lms.b = lab.r - 0.0894841775 * lab.g - 1.2914855480 * lab.b;
    lms = sign(lms) * pow(abs(lms), vec3(3.0));
    vec3 rgb;
    rgb.r =  4.0767416621 * lms.r - 3.3077115913 * lms.g + 0.2309699292 * lms.b;
    rgb.g = -1.2684380046 * lms.r + 2.6097574011 * lms.g - 0.3413193965 * lms.b;
    rgb.b = -0.0041960863 * lms.r - 0.7034186147 * lms.g + 1.7076147010 * lms.b;
    return rgb;
}
vec3 srgbToOkLab(vec3 srgb) {
    return rgbToOkLab(srgbToLinearVec3(srgb));
}
vec3 okLabToSrgb(vec3 lab) {
    vec3 linear = okLabToRgb(lab);
    linear = clamp(linear, 0.0, 1.0);
    return linearToSrgbVec3(linear);
}

// --- Helpers ---
float totalVel() {
    float tv = 0.0;
    for (int i = 0; i < 5; i++) tv += u_emotionVelocities[i];
    return tv;
}

vec3 getEmotionColor(int idx) {
    if (idx == 0) return u_emotionColors[0];
    if (idx == 1) return u_emotionColors[1];
    if (idx == 2) return u_emotionColors[2];
    if (idx == 3) return u_emotionColors[3];
    return u_emotionColors[4];
}

// --- Motion-vector temporal feedback ---
void applyMotionVectorFeedback(vec2 sampleUV) {
    if (uEnableFeedback > 0.5) {
        vec2 motionVec = vec2(0.0);
        for (int i = 0; i < 5; i++) {
            float angle = float(i) * 1.2566; // 2*PI/5
            motionVec += vec2(cos(angle), sin(angle)) * u_emotionVelocities[i];
        }
        motionVec *= 0.01;
        vec4 prevSample = texture(uPrevFrame, sampleUV + motionVec);
        float clampedFeedback = clamp(uFeedbackStrength, 0.0, 0.95);
        fragColor = mix(fragColor, prevSample, clampedFeedback);
    }
}
`;

// ---------------------------------------------------------------------------
// Voronoi helpers shared by modes 0, 1, 2
// ---------------------------------------------------------------------------

const voronoiBlock = `
    // Voronoi cellular pattern
    float slowTime = u_time * 0.05;
    vec2 cellSize = vec2(0.12, 0.25);
    vec2 cellCoord = floor(warpedUV / cellSize);

    float minDist = 10.0;
    float secondDist = 10.0;
    int winnerEmotion = 0;
    int secondEmotion = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 neighbor = cellCoord + vec2(float(dx), float(dy));
            vec2 seedOffset = vec2(
                snoise(neighbor + vec2(slowTime * 0.3, 0.0)) * 0.4 + 0.5,
                snoise(neighbor + vec2(0.0, slowTime * 0.2 + 100.0)) * 0.4 + 0.5
            );
            float cellHash = snoise(neighbor * 7.3 + 50.0);
            float cumulative = 0.0;
            int emotion = 4;
            float threshold = cellHash * 0.5 + 0.5;
            cumulative += u_emotionValues[0];
            if (threshold < cumulative) emotion = 0;
            else {
                cumulative += u_emotionValues[1];
                if (threshold < cumulative) emotion = 1;
                else {
                    cumulative += u_emotionValues[2];
                    if (threshold < cumulative) emotion = 2;
                    else {
                        cumulative += u_emotionValues[3];
                        if (threshold < cumulative) emotion = 3;
                    }
                }
            }
            vec2 seedPos = (neighbor + seedOffset) * cellSize;
            float dist = length(warpedUV - seedPos);
            if (dist < minDist) {
                secondDist = minDist;
                secondEmotion = winnerEmotion;
                minDist = dist;
                winnerEmotion = emotion;
            } else if (dist < secondDist) {
                secondDist = dist;
                secondEmotion = emotion;
            }
        }
    }

    vec3 winnerColor = getEmotionColor(winnerEmotion);
    vec3 secondColor = getEmotionColor(secondEmotion);
    float edgeDist = secondDist - minDist;
    float t = smoothstep(0.0, 0.02, edgeDist);
    vec3 blendedColor = mix(secondColor, winnerColor, t);
`;

// ---------------------------------------------------------------------------
// Mode 0: Voronoi (original baseline)
// ---------------------------------------------------------------------------

const mode0Main = `
void main() {
    float tv = totalVel();
    float noiseScale = 0.5 + tv * 1.5;
    float distortionStrength = tv * 0.03;
    vec2 noiseOffset1 = vec2(
        snoise(v_uv * noiseScale + u_time * 0.2),
        snoise(v_uv * noiseScale + u_time * 0.2 + 100.0)
    );
    vec2 noiseOffset2 = vec2(
        snoise(v_uv * noiseScale * 2.0 + u_time * 0.3 + 50.0),
        snoise(v_uv * noiseScale * 2.0 + u_time * 0.3 + 150.0)
    ) * 0.5;
    vec2 warpedUV = v_uv + (noiseOffset1 + noiseOffset2) * distortionStrength;

    vec4 maskSample = texture(u_asciiMask, warpedUV);
    float maskAlpha = maskSample.a;

    ${voronoiBlock}

    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(warpedUV);
}`;

// ---------------------------------------------------------------------------
// Mode 1: Curl Noise distortion + Voronoi pattern
// ---------------------------------------------------------------------------

const mode1Main = `
void main() {
    float tv = totalVel();
    float noiseScale = 0.5 + tv * 1.5;
    float distortionStrength = tv * 0.03;

    // Curl noise produces divergence-free swirling flow
    vec2 flow = curlNoise(v_uv * noiseScale, u_time) * distortionStrength;
    vec2 warpedUV = v_uv + flow;

    vec4 maskSample = texture(u_asciiMask, warpedUV);
    float maskAlpha = maskSample.a;

    ${voronoiBlock}

    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(warpedUV);
}`;

// ---------------------------------------------------------------------------
// Mode 2: Domain-Warped FBM distortion + Voronoi pattern
// ---------------------------------------------------------------------------

const mode2Main = `
void main() {
    float tv = totalVel();
    float noiseScale = 0.5 + tv * 1.5;
    float distortionStrength = tv * 0.03;

    // Double domain warping: warp input to FBM using another FBM
    vec2 warpedP = v_uv + vec2(
        fbm(v_uv * noiseScale + vec2(0.0, 0.0), u_time),
        fbm(v_uv * noiseScale + vec2(5.2, 1.3), u_time)
    ) * 0.15;
    vec2 finalOffset = vec2(
        fbm(warpedP * noiseScale + vec2(1.7, 9.2), u_time),
        fbm(warpedP * noiseScale + vec2(8.3, 2.8), u_time)
    ) * distortionStrength;
    vec2 warpedUV = v_uv + finalOffset;

    vec4 maskSample = texture(u_asciiMask, warpedUV);
    float maskAlpha = maskSample.a;

    ${voronoiBlock}

    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(warpedUV);
}`;

// ---------------------------------------------------------------------------
// Mode 3: Metaballs (implicit surfaces) + simplex distortion
// ---------------------------------------------------------------------------

const mode3Main = `
void main() {
    float tv = totalVel();
    float slowTime = u_time * 0.05;

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;

    // Metaball field: 1 ball per emotion (5 total)
    float totalField = 0.0;
    vec3 colorAccum = vec3(0.0);

    for (int i = 0; i < 5; i++) {
        vec2 center = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.4 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.4 + 0.5
        );
        float radius = max(u_emotionValues[i], 0.1) * 0.3 + 0.05;
        float dist = length(v_uv - center);
        float field = radius * radius / (dist * dist + 0.001);
        totalField += field;
        colorAccum += getEmotionColor(i) * field;
    }
    vec3 blendedColor = colorAccum / max(totalField, 0.001);
    float metaAlpha = smoothstep(0.8, 1.2, totalField);

    fragColor = vec4(blendedColor * maskAlpha * metaAlpha, maskAlpha * metaAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 4: Flow Field Advection (replaces both distortion and spatial pattern)
// ---------------------------------------------------------------------------

const mode4Main = `
void main() {
    float tv = totalVel();
    float slowTime = u_time * 0.05;

    // Backward advection: look up where this pixel's color came from
    vec2 vel = curlNoise(v_uv * 3.0, u_time) * tv * 0.02;
    vec2 advectedUV = v_uv - vel;
    vec4 advected = texture(uPrevFrame, advectedUV);

    // Color injection at drifting source points per emotion
    float injection = 0.0;
    vec3 injectedColor = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float strength = u_emotionValues[i];
        float d = length(v_uv - source);
        float contrib = strength * smoothstep(0.12, 0.0, d);
        injection += contrib;
        injectedColor += getEmotionColor(i) * contrib;
    }

    // Blend advected previous frame with fresh injection
    vec3 finalColor;
    if (injection > 0.001) {
        vec3 normInjected = injectedColor / injection;
        finalColor = mix(advected.rgb * 0.985, normInjected, min(injection, 1.0));
    } else {
        finalColor = advected.rgb * 0.985; // Slow decay
    }

    // Apply ASCII mask
    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(finalColor * maskAlpha, maskAlpha);

    // Motion-vector feedback (lighter since advection already provides persistence)
    if (uEnableFeedback > 0.5) {
        vec2 motionVec = vec2(0.0);
        for (int i = 0; i < 5; i++) {
            float angle = float(i) * 1.2566;
            motionVec += vec2(cos(angle), sin(angle)) * u_emotionVelocities[i];
        }
        motionVec *= 0.005; // Lighter than other modes
        vec4 prevSample = texture(uPrevFrame, v_uv + motionVec);
        float clampedFeedback = clamp(uFeedbackStrength * 0.5, 0.0, 0.5);
        fragColor = mix(fragColor, prevSample, clampedFeedback);
    }
}`;

// ---------------------------------------------------------------------------
// Mode 5: Reaction-Diffusion render pass (reads R-D state texture)
// ---------------------------------------------------------------------------

const mode5Main = `
uniform sampler2D uRDState; // R=U, G=V from simulation

void main() {
    float tv = totalVel();
    float slowTime = u_time * 0.05;

    // Read R-D simulation state
    vec4 rdState = texture(uRDState, v_uv);
    float V = rdState.g;

    // Map V concentration to emotion colors based on nearest source
    float totalWeight = 0.0;
    vec3 colorAccum = vec3(0.0);
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float d = length(v_uv - source);
        float w = max(u_emotionValues[i], 0.1) / (d * d + 0.01);
        totalWeight += w;
        colorAccum += getEmotionColor(i) * w;
    }
    vec3 baseColor = colorAccum / max(totalWeight, 0.001);

    // V concentration modulates visibility — lower threshold for idle visibility
    float pattern = smoothstep(0.05, 0.2, V);

    // Apply ASCII mask
    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(baseColor * pattern * maskAlpha, pattern * maskAlpha);

    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 6: Chladni Resonance Patterns
// ---------------------------------------------------------------------------

const mode6Main = `
void main() {
    float tv = totalVel();
    vec2 p = v_uv * 2.0 - 1.0;

    // 5 mode pairs: (n, m) per emotion
    // Serene: (2,1), Vibrant: (5,4), Melancholy: (3,1), Curious: (4,3), Content: (3,3)
    float modeN[5];
    float modeM[5];
    modeN[0] = 2.0; modeM[0] = 1.0;
    modeN[1] = 5.0; modeM[1] = 4.0;
    modeN[2] = 3.0; modeM[2] = 1.0;
    modeN[3] = 4.0; modeM[3] = 3.0;
    modeN[4] = 3.0; modeM[4] = 3.0;

    float field = 0.0;
    float totalAmp = 0.0;
    vec3 colorAccum = vec3(0.0);
    float PI = 3.14159265;

    for (int i = 0; i < 5; i++) {
        float amp = max(u_emotionValues[i], 0.1);
        float phi = u_emotionVelocities[i] * u_time * 2.0;
        float n = modeN[i];
        float m = modeM[i];
        // Chladni eigenmode: cos(n*PI*x)*cos(m*PI*y) - cos(m*PI*x)*cos(n*PI*y)
        float chladni = cos(n * PI * p.x + phi) * cos(m * PI * p.y)
                      - cos(m * PI * p.x + phi) * cos(n * PI * p.y);
        field += amp * chladni;
        totalAmp += amp;
        vec3 ec = getEmotionColor(i);
        vec3 eLab = srgbToOkLab(ec);
        colorAccum += eLab * amp * abs(chladni);
    }

    field /= max(totalAmp, 0.001);
    colorAccum /= max(totalAmp, 0.001);

    // Nodal line width controlled by velocity
    float lineBlur = 0.02 + tv * 0.06;
    float edgeFactor = smoothstep(lineBlur, lineBlur * 2.0, abs(field));

    vec3 lab = colorAccum;
    lab.x = 0.3 + edgeFactor * 0.5;
    vec3 blendedColor = okLabToSrgb(lab);

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 7: Cymatics / Wave Interference
// ---------------------------------------------------------------------------

const mode7Main = `
void main() {
    float tv = totalVel();
    float slowTime = u_time * 0.05;

    float field = 0.0;
    vec3 colorAccum = vec3(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < 5; i++) {
        float amp = max(u_emotionValues[i], 0.1);

        // Drifting source positions
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );

        float k = 20.0 + u_emotionVelocities[i] * 80.0;
        float omega = 2.0 + amp * 4.0;
        float dist = length(v_uv - source);
        float wave = sin(k * dist - omega * u_time);
        float contrib = amp * wave;

        field += contrib;
        colorAccum += getEmotionColor(i) * amp * max(wave, 0.0);
        totalWeight += amp * max(wave, 0.0);
    }

    field = field / max(totalWeight, 0.001) * 0.5 + 0.5;

    vec3 blendedColor;
    if (totalWeight > 0.001) {
        vec3 normColor = colorAccum / totalWeight;
        vec3 lab = srgbToOkLab(normColor);
        lab.x = 0.2 + field * 0.6;
        blendedColor = okLabToSrgb(lab);
    } else {
        blendedColor = vec3(0.05);
    }

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 8: Julia Set Morphing
// ---------------------------------------------------------------------------

const mode8Main = `
void main() {
    float tv = totalVel();

    // 5 preset c-values blended by emotion values
    vec2 cPresets[5];
    cPresets[0] = vec2(-0.4, 0.6);
    cPresets[1] = vec2(0.285, 0.01);
    cPresets[2] = vec2(-0.8, 0.156);
    cPresets[3] = vec2(-0.7269, 0.1889);
    cPresets[4] = vec2(0.0, 0.8);

    // Blend c-values with floor so Julia set is always interesting
    vec2 cBlend = vec2(0.0);
    float totalVal = 0.0;
    for (int i = 0; i < 5; i++) {
        float v = max(u_emotionValues[i], 0.1);
        cBlend += cPresets[i] * v;
        totalVal += v;
    }
    cBlend /= totalVal;

    // Zoom breathing from velocities
    float zoom = 1.5 + sin(u_time * 0.5) * tv * 0.3;
    vec2 z = (v_uv * 2.0 - 1.0) * zoom;

    float iter = 0.0;
    for (int i = 0; i < 80; i++) {
        if (dot(z, z) > 4.0) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + cBlend;
        iter += 1.0;
    }

    // Smooth escape time coloring
    float smoothIter = iter;
    if (dot(z, z) > 4.0) {
        smoothIter = iter - log2(log2(dot(z, z))) + 4.0;
    }

    float t = smoothIter / 80.0;
    float angle = atan(z.y, z.x);

    // Weighted emotion color blend — always contributes
    vec3 lab = vec3(0.0);
    float wTotal = 0.0;
    for (int i = 0; i < 5; i++) {
        float w = max(u_emotionValues[i], 0.1) * (0.5 + 0.5 * sin(angle + float(i) * 1.2566));
        w = max(w, 0.0);
        lab += srgbToOkLab(getEmotionColor(i)) * w;
        wTotal += w;
    }
    lab /= wTotal;

    lab.x = 0.15 + t * 0.6;
    if (iter >= 80.0) lab.x = 0.2;
    vec3 blendedColor = okLabToSrgb(lab);

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 9: Gravitational Lensing
// ---------------------------------------------------------------------------

const mode9Main = `
void main() {
    float tv = totalVel();
    float slowTime = u_time * 0.05;

    vec2 distortedUV = v_uv;
    vec2 lensAccum = vec2(0.0);

    // Emotion mass positions
    for (int i = 0; i < 5; i++) {
        vec2 center = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float mass = max(u_emotionValues[i], 0.1) * 0.05;
        vec2 delta = v_uv - center;
        float dist = length(delta) + 0.001;
        vec2 dir = delta / dist;

        // Gravitational pull
        lensAccum += dir * mass / dist;

        // Frame-dragging from velocity (Kerr rotation)
        float frameDrag = u_emotionVelocities[i] * 0.02;
        lensAccum += vec2(-dir.y, dir.x) * frameDrag / (dist * dist);
    }

    distortedUV += lensAccum;
    distortedUV = clamp(distortedUV, 0.0, 1.0);

    // Caustic brightness via Jacobian (finite differences)
    float eps = 0.002;
    float tv2 = totalVel();
    vec2 du = vec2(eps, 0.0);
    vec2 dv = vec2(0.0, eps);

    vec2 lensR = vec2(0.0), lensU = vec2(0.0), lensV = vec2(0.0);
    for (int i = 0; i < 5; i++) {
        vec2 center = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float mass = max(u_emotionValues[i], 0.1) * 0.05;
        vec2 deltaR = (v_uv + du) - center;
        vec2 deltaU = (v_uv + dv) - center;
        float distR = length(deltaR) + 0.001;
        float distU = length(deltaU) + 0.001;
        lensR += deltaR / (distR * distR * distR) * mass;
        lensU += deltaU / (distU * distU * distU) * mass;
    }

    vec2 col1 = v_uv + du + lensR;
    vec2 col2 = v_uv + dv + lensU;
    // Approximate Jacobian determinant for caustic brightness
    float det = abs((col1.x - distortedUV.x) * (col2.y - distortedUV.y)
                  - (col1.y - distortedUV.y) * (col2.x - distortedUV.x));
    det = max(det, 0.001);
    float caustic = 1.0 / det;
    caustic = clamp(caustic, 0.5, 3.0);

    // Sample previous frame at distorted UV for lensed geometry
    vec4 prevSample = texture(uPrevFrame, distortedUV);

    // Emotion-colored background gradient
    vec3 bgLab = vec3(0.0);
    float bgW = 0.0;
    for (int i = 0; i < 5; i++) {
        float bv = max(u_emotionValues[i], 0.1);
        bgLab += srgbToOkLab(getEmotionColor(i)) * bv;
        bgW += bv;
    }
    bgLab /= bgW;
    bgLab.x = 0.3;
    vec3 bgColor = okLabToSrgb(bgLab) * caustic;

    vec3 finalColor = mix(bgColor, prevSample.rgb, 0.5);

    vec4 maskSample = texture(u_asciiMask, distortedUV);
    float maskAlpha = maskSample.a;
    fragColor = vec4(finalColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(distortedUV);
}`;

// ---------------------------------------------------------------------------
// Mode 10: Strange Attractor (Clifford)
// ---------------------------------------------------------------------------

const mode10Main = `
void main() {
    float tv = totalVel();

    // Clifford attractor parameters from emotion values + temporal drift
    float a = mix(-2.0, 2.0, u_emotionValues[0]) + sin(u_time * 0.3) * u_emotionVelocities[0] * 0.5;
    float b = mix(-2.0, 2.0, u_emotionValues[1]) + cos(u_time * 0.2) * u_emotionVelocities[1] * 0.5;
    float c = mix(-1.5, 1.5, u_emotionValues[2]) + sin(u_time * 0.4) * u_emotionVelocities[2] * 0.3;
    float d = mix(-1.5, 1.5, u_emotionValues[3]) + cos(u_time * 0.35) * u_emotionVelocities[3] * 0.3;
    float scale = 1.0 + u_emotionValues[4] * 2.0;

    vec2 p = (v_uv * 2.0 - 1.0) / scale;

    // 6 iterations of the Clifford map
    for (int iter = 0; iter < 6; iter++) {
        float xNew = sin(a * p.y) + c * cos(a * p.x);
        float yNew = sin(b * p.x) + d * cos(b * p.y);
        p = vec2(xNew, yNew);
    }

    // Map final position to color
    float angle = atan(p.y, p.x);
    vec3 lab = vec3(0.0);
    float wTotal = 0.0;
    for (int i = 0; i < 5; i++) {
        if (u_emotionValues[i] < 0.001) continue;
        float w = u_emotionValues[i] * (0.5 + 0.5 * cos(angle - float(i) * 1.2566));
        w = max(w, 0.0);
        lab += srgbToOkLab(getEmotionColor(i)) * w;
        wTotal += w;
    }
    if (wTotal > 0.001) lab /= wTotal;

    // Distance from origin → brightness (convergence zones bright)
    float dist = length(p);
    lab.x = smoothstep(2.0, 0.0, dist) * 0.7 + 0.1;
    vec3 blendedColor = okLabToSrgb(lab);

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 11: Voronoi Crack Propagation
// ---------------------------------------------------------------------------

const mode11Main = `
void main() {
    float tv = totalVel();
    float noiseScale = 0.5 + tv * 1.5;
    float distortionStrength = tv * 0.03;

    vec2 noiseOffset1 = vec2(
        snoise(v_uv * noiseScale + u_time * 0.2),
        snoise(v_uv * noiseScale + u_time * 0.2 + 100.0)
    );
    vec2 noiseOffset2 = vec2(
        snoise(v_uv * noiseScale * 2.0 + u_time * 0.3 + 50.0),
        snoise(v_uv * noiseScale * 2.0 + u_time * 0.3 + 150.0)
    ) * 0.5;
    vec2 warpedUV = v_uv + (noiseOffset1 + noiseOffset2) * distortionStrength;

    vec4 maskSample = texture(u_asciiMask, warpedUV);
    float maskAlpha = maskSample.a;

    float slowTime = u_time * 0.05;
    vec2 cellSize = vec2(0.12, 0.25);
    vec2 cellCoord = floor(warpedUV / cellSize);

    float minDist = 10.0;
    float secondDist = 10.0;
    int winnerEmotion = 0;
    int secondEmotion = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 neighbor = cellCoord + vec2(float(dx), float(dy));
            vec2 seedOffset = vec2(
                snoise(neighbor + vec2(slowTime * 0.3, 0.0)) * 0.4 + 0.5,
                snoise(neighbor + vec2(0.0, slowTime * 0.2 + 100.0)) * 0.4 + 0.5
            );
            float cellHash = snoise(neighbor * 7.3 + 50.0);
            float cumulative = 0.0;
            int emotion = 4;
            float threshold = cellHash * 0.5 + 0.5;
            cumulative += u_emotionValues[0];
            if (threshold < cumulative) emotion = 0;
            else {
                cumulative += u_emotionValues[1];
                if (threshold < cumulative) emotion = 1;
                else {
                    cumulative += u_emotionValues[2];
                    if (threshold < cumulative) emotion = 2;
                    else {
                        cumulative += u_emotionValues[3];
                        if (threshold < cumulative) emotion = 3;
                    }
                }
            }
            vec2 seedPos = (neighbor + seedOffset) * cellSize;
            float dist = length(warpedUV - seedPos);
            if (dist < minDist) {
                secondDist = minDist;
                secondEmotion = winnerEmotion;
                minDist = dist;
                winnerEmotion = emotion;
            } else if (dist < secondDist) {
                secondDist = dist;
                secondEmotion = emotion;
            }
        }
    }

    // Dynamic crack width from total velocity
    float crackWidth = tv * 0.15;
    float edgeDist = secondDist - minDist;

    // Noise-perturbed edges
    float noisePerturb = snoise(warpedUV * 20.0) * tv * 0.02;
    edgeDist += noisePerturb;

    float crackFactor = smoothstep(crackWidth, crackWidth * 0.3, edgeDist);

    // Cell interior color (standard emotion-weighted OKLab blend)
    vec3 winnerColor = getEmotionColor(winnerEmotion);
    vec3 cellLab = srgbToOkLab(winnerColor);
    cellLab.x = 0.35;

    // Crack color: complement (negate a, b in OKLab)
    vec3 crackLab = cellLab;
    crackLab.g = -crackLab.g;
    crackLab.b = -crackLab.b;
    crackLab.x = 0.7;

    vec3 finalLab = mix(cellLab, crackLab, crackFactor);
    vec3 blendedColor = okLabToSrgb(finalLab);

    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(warpedUV);
}`;

// ---------------------------------------------------------------------------
// Mode 12: Smoke render pass (reads simulation state)
// ---------------------------------------------------------------------------

const mode12Main = `
uniform sampler2D uSmokeState;

void main() {
    vec4 smokeState = texture(uSmokeState, v_uv);
    float density = smokeState.r;
    float temperature = smokeState.g;

    // Color by emotion source proximity (reuse pattern from other modes)
    float slowTime = u_time * 0.05;
    vec3 colorAccum = vec3(0.0);
    float wTotal = 0.0;
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float d = length(v_uv - source);
        float w = u_emotionValues[i] / (d * d + 0.01);
        wTotal += w;
        colorAccum += getEmotionColor(i) * w;
    }
    vec3 baseColor = colorAccum / max(wTotal, 0.001);

    vec3 lab = srgbToOkLab(baseColor);
    lab.x = 0.15 + density * 0.6;
    // Temperature boosts saturation
    lab.g *= 1.0 + temperature * 0.5;
    lab.b *= 1.0 + temperature * 0.5;
    vec3 blendedColor = okLabToSrgb(lab);

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    float alpha = density * maskAlpha;
    fragColor = vec4(blendedColor * alpha, alpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 13: Topography render pass (reads erosion state)
// ---------------------------------------------------------------------------

const mode13Main = `
uniform sampler2D uErosionState;

void main() {
    vec4 erosionState = texture(uErosionState, v_uv);
    float height = erosionState.r;
    float water = erosionState.g;

    // Contour lines: detect zero crossings of fract(height * count)
    float contourCount = 20.0;
    float contour = abs(fract(height * contourCount) - 0.5) * 2.0;
    float contourLine = 1.0 - smoothstep(0.0, 0.1, contour);

    // Slope shading (gradient magnitude via finite differences)
    float texel = 1.0 / 256.0;
    float hN = texture(uErosionState, v_uv + vec2(0.0, texel)).r;
    float hS = texture(uErosionState, v_uv - vec2(0.0, texel)).r;
    float hE = texture(uErosionState, v_uv + vec2(texel, 0.0)).r;
    float hW = texture(uErosionState, v_uv - vec2(texel, 0.0)).r;
    float slope = length(vec2(hE - hW, hN - hS)) / (2.0 * texel);
    float slopeDarken = 1.0 - clamp(slope * 0.5, 0.0, 0.6);

    // Color by height mapped through emotion palette
    vec3 lab = vec3(0.0);
    float wTotal = 0.0;
    for (int i = 0; i < 5; i++) {
        if (u_emotionValues[i] < 0.001) continue;
        float w = u_emotionValues[i] * (0.5 + 0.5 * sin(height * 6.28 + float(i) * 1.2566));
        w = max(w, 0.0);
        lab += srgbToOkLab(getEmotionColor(i)) * w;
        wTotal += w;
    }
    if (wTotal > 0.001) lab /= wTotal;

    lab.x = 0.2 + height * 0.5;
    // Water channels: slight blue tint
    if (water > 0.1) {
        lab.b += water * 0.1;
    }

    vec3 blendedColor = okLabToSrgb(lab) * slopeDarken;
    // Contour lines darken
    blendedColor *= 1.0 - contourLine * 0.5;

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 14: Magnetic Field Lines (Progressive LIC)
// ---------------------------------------------------------------------------

const mode14Main = `
void main() {
    float tv = totalVel();
    float slowTime = u_time * 0.05;

    // 5 magnetic dipoles
    vec3 licColor = vec3(0.0);
    vec2 fieldDir = vec2(0.0);
    float fieldMag = 0.0;

    for (int i = 0; i < 5; i++) {
        float strength = u_emotionValues[i];
        if (strength < 0.001) continue;

        vec2 center = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );

        // Dipole orientation rotates with velocity
        float orientAngle = u_emotionVelocities[i] * u_time * 2.0 + float(i) * 1.2566;
        vec2 m = vec2(cos(orientAngle), sin(orientAngle)) * strength;

        vec2 r = v_uv - center;
        float rLen = length(r) + 0.001;
        vec2 rHat = r / rLen;
        // Dipole field: B = (3(m.r_hat)r_hat - m) / |r|^3
        vec2 B = (3.0 * dot(m, rHat) * rHat - m) / (rLen * rLen * rLen);

        fieldDir += B * strength;
        fieldMag += length(B) * strength;

        licColor += getEmotionColor(i) * strength;
    }

    if (fieldMag < 0.001) {
        vec4 maskSample = texture(u_asciiMask, v_uv);
        fragColor = vec4(vec3(0.02) * maskSample.a, maskSample.a);
        return;
    }

    fieldDir /= fieldMag;
    float fLen = length(fieldDir) + 0.001;
    fieldDir /= fLen;

    // Progressive LIC: 8 samples along field direction
    vec3 accum = vec3(0.0);
    float stepSize = 0.003;
    for (int j = -4; j <= 4; j++) {
        if (j == 0) continue;
        vec2 sampleUV = v_uv + fieldDir * stepSize * float(j);
        sampleUV = clamp(sampleUV, 0.0, 1.0);
        vec4 prev = texture(uPrevFrame, sampleUV);
        accum += prev.rgb;
    }
    accum /= 8.0;

    // Inject fresh color at source points
    vec3 injectColor = licColor / max(fieldMag, 0.001);
    vec3 freshColor = srgbToOkLab(injectColor) * 0.05;

    // Blend accumulated LIC with fresh injection
    vec3 prevLab = srgbToOkLab(accum);
    vec3 finalLab = prevLab + freshColor;
    finalLab = clamp(finalLab, 0.0, 1.0);
    vec3 blendedColor = okLabToSrgb(finalLab);

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 15: Lissajous / Oscilloscope Harmonics
// ---------------------------------------------------------------------------

const mode15Main = `
void main() {
    float tv = totalVel();

    // Lissajous parameters per emotion
    float A[5];
    float fX[5];
    float fY[5];
    float phiX[5];
    float phiY[5];

    for (int i = 0; i < 5; i++) {
        A[i] = u_emotionValues[i];
        fX[i] = (1.0 + float(i) * 0.7) * (1.0 + u_emotionVelocities[i] * 5.0);
        fY[i] = (1.0 + float(i) * 0.5 + 0.3) * (1.0 + u_emotionVelocities[i] * 3.0);
        phiX[i] = float(i) * 1.047;
        phiY[i] = float(i) * 0.785;
    }

    // Evaluate curve at 30 parameter values, find min distance
    float minDist = 10.0;
    int closestEmotion = 0;

    for (int k = 0; k < 30; k++) {
        float t = float(k) / 30.0 * 6.2832 + u_time * 0.5;

        for (int i = 0; i < 5; i++) {
            if (A[i] < 0.01) continue;
            float cx = 0.0;
            float cy = 0.0;
            for (int j = 0; j < 5; j++) {
                if (A[j] < 0.01) continue;
                cx += A[j] * sin(fX[j] * t + phiX[j]);
                cy += A[j] * cos(fY[j] * t + phiY[j]);
            }
            cx *= 0.2;
            cy *= 0.2;

            vec2 curvePos = vec2(cx + 0.5, cy + 0.5);
            float d = length(v_uv - curvePos);
            if (d < minDist) {
                minDist = d;
                closestEmotion = i;
            }
        }
    }

    // Glow rendering
    float glow = 1.0 / (minDist * 150.0 + 0.01);
    glow = clamp(glow, 0.0, 1.0);

    vec3 curveColor = srgbToOkLab(getEmotionColor(closestEmotion));
    curveColor.x = glow * 0.8;

    // Trail via previous frame with decay
    vec4 prevSample = texture(uPrevFrame, v_uv);
    vec3 prevLab = srgbToOkLab(prevSample.rgb) * 0.97;

    vec3 finalLab = max(prevLab, curveColor * glow);
    vec3 blendedColor = okLabToSrgb(clamp(finalLab, 0.0, 1.0));

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    float alpha = max(glow * 0.8, length(prevSample.rgb) * 0.3);
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha * min(alpha + 0.2, 1.0));

    // Lighter feedback since trails are built-in
    if (uEnableFeedback > 0.5) {
        vec2 motionVec = vec2(0.0);
        for (int i = 0; i < 5; i++) {
            float angle = float(i) * 1.2566;
            motionVec += vec2(cos(angle), sin(angle)) * u_emotionVelocities[i];
        }
        motionVec *= 0.003;
        vec4 prev = texture(uPrevFrame, v_uv + motionVec);
        float fb = clamp(uFeedbackStrength * 0.3, 0.0, 0.3);
        fragColor = mix(fragColor, prev, fb);
    }
}`;

// ---------------------------------------------------------------------------
// Mode 16: Phyllotaxis / Fibonacci Spiral Growth
// ---------------------------------------------------------------------------

const mode16Main = `
void main() {
    float tv = totalVel();

    float goldenAngle = 2.399963; // 137.508 degrees in radians
    vec3 colorAccum = vec3(0.0);
    float totalGlow = 0.0;

    // 5 growth centers
    for (int c = 0; c < 5; c++) {
        if (u_emotionValues[c] < 0.001) continue;

        vec2 center = vec2(
            0.2 + 0.15 * cos(float(c) * 1.2566),
            0.2 + 0.15 * sin(float(c) * 1.2566)
        );

        float maxN = u_emotionValues[c] * 200.0;
        float divergence = goldenAngle + u_emotionVelocities[c] * 0.087; // ~5 degrees max perturbation
        float spacing = 0.012;

        vec2 p = v_uv - center;

        // Find nearest phyllotaxis point
        float r = length(p);
        float angle = atan(p.y, p.x);

        float nApprox = r * r / (spacing * spacing);
        float bestDist = 10.0;

        for (int di = -1; di <= 1; di++) {
            float n = floor(nApprox) + float(di);
            if (n < 0.0 || n > maxN) continue;

            float theta = n * divergence;
            float px = sqrt(n) * spacing * cos(theta);
            float py = sqrt(n) * spacing * sin(theta);
            float d = length(p - vec2(px, py));
            bestDist = min(bestDist, d);
        }

        float pointRadius = 0.005 + u_emotionValues[c] * 0.003;
        float glow = 1.0 - smoothstep(pointRadius * 0.5, pointRadius * 1.5, bestDist);

        vec3 ec = getEmotionColor(c);
        colorAccum += srgbToOkLab(ec) * glow * u_emotionValues[c];
        totalGlow += glow * u_emotionValues[c];
    }

    vec3 lab;
    if (totalGlow > 0.001) {
        lab = colorAccum / totalGlow;
        lab.x = min(totalGlow * 0.8, 0.8);
    } else {
        lab = vec3(0.05);
    }
    vec3 blendedColor = okLabToSrgb(lab);

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Mode 17: Lattice Boltzmann (Ink Flow) render pass
// ---------------------------------------------------------------------------

const mode17Main = `
uniform sampler2D uLBMState;

void main() {
    // LBM state packed: tex0.rgba = f0,f1,f2,f3  tex1.rgba = f4,f5,f6,f7  tex2.rg = f8,density
    // We read from a combined render — the renderer will bind the density+velocity texture
    vec4 state0 = texture(uLBMState, v_uv);

    // For the render pass, we use a simpler visualization texture (density in R, velX in G, velY in B)
    float density = state0.r;
    float velX = state0.g;
    float velY = state0.b;
    float vorticity = abs(velX - velY);

    // Color by velocity direction mapped through emotion palette
    float velAngle = atan(velY, velX);
    vec3 lab = vec3(0.0);
    float wTotal = 0.0;
    for (int i = 0; i < 5; i++) {
        if (u_emotionValues[i] < 0.001) continue;
        float w = u_emotionValues[i] * (0.5 + 0.5 * cos(velAngle - float(i) * 1.2566));
        w = max(w, 0.0);
        lab += srgbToOkLab(getEmotionColor(i)) * w;
        wTotal += w;
    }
    if (wTotal > 0.001) lab /= wTotal;

    lab.x = clamp(density * 0.8, 0.05, 0.85);
    // Vorticity at vortex cores → color intensity boost
    lab.g *= 1.0 + vorticity * 2.0;
    lab.b *= 1.0 + vorticity * 2.0;

    vec3 blendedColor = okLabToSrgb(clamp(lab, 0.0, 1.0));

    vec4 maskSample = texture(u_asciiMask, v_uv);
    float maskAlpha = maskSample.a;
    fragColor = vec4(blendedColor * maskAlpha, maskAlpha);
    applyMotionVectorFeedback(v_uv);
}`;

// ---------------------------------------------------------------------------
// Assemble complete fragment shader sources
// ---------------------------------------------------------------------------

export const fragmentShaderSources: string[] = [
    shaderPreamble + mode0Main,
    shaderPreamble + mode1Main,
    shaderPreamble + mode2Main,
    shaderPreamble + mode3Main,
    shaderPreamble + mode4Main,
    shaderPreamble + mode5Main,
    shaderPreamble + mode6Main,
    shaderPreamble + mode7Main,
    shaderPreamble + mode8Main,
    shaderPreamble + mode9Main,
    shaderPreamble + mode10Main,
    shaderPreamble + mode11Main,
    shaderPreamble + mode12Main,
    shaderPreamble + mode13Main,
    shaderPreamble + mode14Main,
    shaderPreamble + mode15Main,
    shaderPreamble + mode16Main,
    shaderPreamble + mode17Main,
];

// Keep legacy export for compatibility
export const emotionFieldFragmentShaderSource = fragmentShaderSources[0];

// ---------------------------------------------------------------------------
// Reaction-Diffusion simulation shader (separate program)
// ---------------------------------------------------------------------------

export const rdSimulationFragmentSource = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D uState;      // Current R-D state (R=U, G=V)
uniform vec2 uResolution;      // Texture resolution
uniform float u_time;
uniform float u_emotionValues[5];
out vec4 fragColor;

// Simplex noise (duplicated for standalone shader)
vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                      + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

void main() {
    // Gray-Scott parameters (labyrinthine/coral patterns)
    const float Du = 1.0;
    const float Dv = 0.5;
    const float F = 0.055;
    const float k = 0.062;
    const float dt = 1.0;

    vec4 state = texture(uState, v_uv);
    float U = state.r;
    float V = state.g;

    // 5-point Laplacian
    vec2 texel = 1.0 / uResolution;
    float U_n = texture(uState, v_uv + vec2(0.0, texel.y)).r;
    float U_s = texture(uState, v_uv - vec2(0.0, texel.y)).r;
    float U_e = texture(uState, v_uv + vec2(texel.x, 0.0)).r;
    float U_w = texture(uState, v_uv - vec2(texel.x, 0.0)).r;
    float lapU = U_n + U_s + U_e + U_w - 4.0 * U;

    float V_n = texture(uState, v_uv + vec2(0.0, texel.y)).g;
    float V_s = texture(uState, v_uv - vec2(0.0, texel.y)).g;
    float V_e = texture(uState, v_uv + vec2(texel.x, 0.0)).g;
    float V_w = texture(uState, v_uv - vec2(texel.x, 0.0)).g;
    float lapV = V_n + V_s + V_e + V_w - 4.0 * V;

    // Gray-Scott equations
    float UVV = U * V * V;
    float newU = U + (Du * lapU - UVV + F * (1.0 - U)) * dt;
    float newV = V + (Dv * lapV + UVV - (F + k) * V) * dt;

    // Inject V at emotion source points
    float slowTime = u_time * 0.05;
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float dist = length(v_uv - source);
        float inject = max(u_emotionValues[i], 0.1) * smoothstep(0.05, 0.0, dist) * 0.01;
        newV += inject;
    }

    fragColor = vec4(clamp(newU, 0.0, 1.0), clamp(newV, 0.0, 1.0), 0.0, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Smoke simulation shader (semi-Lagrangian advection + buoyancy)
// ---------------------------------------------------------------------------

export const smokeSimulationFragmentSource = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D uState;
uniform vec2 uResolution;
uniform float u_time;
uniform float u_emotionValues[5];
uniform float u_emotionVelocities[5];
out vec4 fragColor;

vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                      + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

void main() {
    vec4 state = texture(uState, v_uv);
    float density = state.r;
    float temperature = state.g;
    float velX = state.b;
    float velY = state.a;

    float dt = 0.1;
    float slowTime = u_time * 0.05;

    // Semi-Lagrangian advection: backward trace
    vec2 advUV = v_uv - vec2(velX, velY) * dt;
    advUV = clamp(advUV, 0.0, 1.0);
    vec4 advected = texture(uState, advUV);
    density = advected.r;
    temperature = advected.g;

    // Buoyancy: hot smoke rises, cold sinks
    float buoyancy = (temperature - 0.3) * 0.5;
    velY += buoyancy * dt;

    // Curl noise turbulence
    float eps = 0.01;
    float n1 = snoise(vec2(v_uv.x, v_uv.y + eps) + u_time * 0.2);
    float n2 = snoise(vec2(v_uv.x, v_uv.y - eps) + u_time * 0.2);
    float n3 = snoise(vec2(v_uv.x + eps, v_uv.y) + u_time * 0.2);
    float n4 = snoise(vec2(v_uv.x - eps, v_uv.y) + u_time * 0.2);
    velX += (n3 - n4) / (2.0 * eps) * 0.05;
    velY += (n1 - n2) / (2.0 * eps) * 0.05;

    // Inject at emotion source points
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float dist = length(v_uv - source);
        float inject = smoothstep(0.05, 0.0, dist);
        density += u_emotionValues[i] * inject * 0.02;
        temperature += u_emotionVelocities[i] * inject * 0.05;
    }

    // Dissipation
    density *= 0.995;
    temperature *= 0.99;
    velX *= 0.99;
    velY *= 0.99;

    // Clamp all channels
    density = clamp(density, 0.0, 1.0);
    temperature = clamp(temperature, 0.0, 1.0);
    velX = clamp(velX, -1.0, 1.0);
    velY = clamp(velY, -1.0, 1.0);

    fragColor = vec4(density, temperature, velX, velY);
}
`;

// ---------------------------------------------------------------------------
// Erosion simulation shader (hydraulic erosion on heightmap)
// ---------------------------------------------------------------------------

export const erosionSimulationFragmentSource = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D uState;
uniform vec2 uResolution;
uniform float u_time;
uniform float u_emotionValues[5];
uniform float u_emotionVelocities[5];
out vec4 fragColor;

vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                      + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m; m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < 4; i++) {
        value += amplitude * snoise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

void main() {
    vec4 state = texture(uState, v_uv);
    float height = state.r;
    float water = state.g;
    float sediment = state.b;
    float hardness = state.a;

    vec2 texel = 1.0 / uResolution;

    // Height gradient (Sobel-like)
    float hN = texture(uState, v_uv + vec2(0.0, texel.y)).r;
    float hS = texture(uState, v_uv - vec2(0.0, texel.y)).r;
    float hE = texture(uState, v_uv + vec2(texel.x, 0.0)).r;
    float hW = texture(uState, v_uv - vec2(texel.x, 0.0)).r;
    vec2 gradient = vec2(hE - hW, hN - hS) * 0.5;

    // Water flows downhill
    float slope = length(gradient);
    float flowSpeed = slope * 0.5;
    vec2 flowDir = slope > 0.001 ? -gradient / slope : vec2(0.0);

    // Erosion: remove height where water flows fast
    float erosion = flowSpeed * water * 0.001 / max(hardness, 0.1);
    height -= erosion;
    sediment += erosion;

    // Deposition: deposit where water slows
    float deposition = sediment * 0.01 * (1.0 - flowSpeed);
    height += deposition;
    sediment -= deposition;

    // Water redistribution
    water -= water * 0.01; // evaporation

    // Inject water at emotion sources
    float slowTime = u_time * 0.05;
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float dist = length(v_uv - source);
        water += u_emotionValues[i] * smoothstep(0.08, 0.0, dist) * 0.01;
    }

    // Target height from emotion-weighted FBM
    float target = 0.5;
    float totalW = 0.0;
    for (int i = 0; i < 5; i++) {
        float w = u_emotionValues[i];
        if (w < 0.001) continue;
        float n = fbm(v_uv * (2.0 + float(i)) + u_time * 0.02 + float(i) * 5.0);
        if (i == 2) n = -n; // melancholy → valleys
        target += n * w;
        totalW += w;
    }
    if (totalW > 0.001) target = (target / totalW) * 0.5 + 0.5;

    // Lerp height toward target
    height += (target - height) * 0.01;

    height = clamp(height, 0.0, 1.0);
    water = clamp(water, 0.0, 1.0);
    sediment = clamp(sediment, 0.0, 1.0);
    hardness = clamp(hardness, 0.1, 1.0);

    fragColor = vec4(height, water, sediment, hardness);
}
`;

// ---------------------------------------------------------------------------
// LBM simulation shader (D2Q9 Lattice Boltzmann)
// ---------------------------------------------------------------------------

export const lbmSimulationFragmentSource = `#version 300 es
precision highp float;

in vec2 v_uv;
uniform sampler2D uState;
uniform vec2 uResolution;
uniform float u_time;
uniform float u_emotionValues[5];
uniform float u_emotionVelocities[5];
out vec4 fragColor;

vec3 permute(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
                        -0.577350269189626, 0.024390243902439);
    vec2 i = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))
                      + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// D2Q9 lattice velocities and weights
const vec2 e[9] = vec2[9](
    vec2(0.0, 0.0),   // 0: rest
    vec2(1.0, 0.0),    // 1: east
    vec2(0.0, 1.0),    // 2: north
    vec2(-1.0, 0.0),   // 3: west
    vec2(0.0, -1.0),   // 4: south
    vec2(1.0, 1.0),    // 5: NE
    vec2(-1.0, 1.0),   // 6: NW
    vec2(-1.0, -1.0),  // 7: SW
    vec2(1.0, -1.0)    // 8: SE
);
const float w[9] = float[9](4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                               1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0);

float readDist(vec2 uv, vec2 offset, vec2 texel) {
    vec2 sampleUV = clamp(uv + offset * texel, 0.0, 1.0);
    // Pack: we use the R channel of the state texture for density
    // and G,B for velocity. Distributions are stored across multiple channels.
    return texture(uState, sampleUV).r;
}

void main() {
    vec2 texel = 1.0 / uResolution;
    float totalVel = 0.0;
    for (int i = 0; i < 5; i++) totalVel += u_emotionVelocities[i];

    float tau = 0.5 + totalVel * 2.0;
    tau = clamp(tau, 0.51, 2.0);

    // Read current density and velocity from state
    vec4 state = texture(uState, v_uv);
    float rho = state.r;
    vec2 u_vel = vec2(state.g, state.b) * 2.0 - 1.0;

    // BGK collision: compute equilibrium and relax
    float f[9];
    float feq[9];
    float usq = dot(u_vel, u_vel);

    for (int i = 0; i < 9; i++) {
        float eu = dot(e[i], u_vel);
        feq[i] = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
        f[i] = feq[i]; // Simplified: direct equilibrium (works for visualization)
    }

    // Inject emotion sources
    float slowTime = u_time * 0.05;
    for (int i = 0; i < 5; i++) {
        vec2 source = vec2(
            snoise(vec2(float(i) * 3.7, slowTime * 0.4)) * 0.35 + 0.5,
            snoise(vec2(float(i) * 5.1 + 100.0, slowTime * 0.3)) * 0.35 + 0.5
        );
        float dist = length(v_uv - source);
        float inject = u_emotionValues[i] * smoothstep(0.06, 0.0, dist) * 0.1;
        rho += inject;

        // Add velocity from emotion direction
        float angle = float(i) * 1.2566 + u_time * 0.5;
        u_vel += vec2(cos(angle), sin(angle)) * u_emotionVelocities[i] * inject * 2.0;
    }

    // Recompute macroscopic quantities
    rho = 0.0;
    vec2 newU = vec2(0.0);
    for (int i = 0; i < 9; i++) {
        rho += f[i];
        newU += e[i] * f[i];
    }
    rho = max(rho, 0.001);
    u_vel = newU / rho;
    u_vel = clamp(u_vel, -0.3, 0.3);

    // Relaxation toward equilibrium
    usq = dot(u_vel, u_vel);
    for (int i = 0; i < 9; i++) {
        float eu = dot(e[i], u_vel);
        feq[i] = w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
        f[i] += (feq[i] - f[i]) / tau;
    }

    // Store density and velocity for render pass
    float outRho = clamp(rho, 0.0, 1.0);
    vec2 outVel = u_vel * 0.5 + 0.5; // encode [-1,1] -> [0,1]

    // Dissipation
    outRho *= 0.998;

    fragColor = vec4(outRho, outVel.x, outVel.y, 0.0);
}
`;

// ---------------------------------------------------------------------------
// Shader compilation utilities
// ---------------------------------------------------------------------------

export function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error('Failed to create shader');
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(`Failed to compile shader: ${info}`);
    }
    return shader;
}

export function createProgram(gl: WebGL2RenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram {
    const program = gl.createProgram();
    if (!program) throw new Error('Failed to create program');
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(`Failed to link program: ${info}`);
    }
    return program;
}
