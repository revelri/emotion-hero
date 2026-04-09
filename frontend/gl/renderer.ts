/**
 * WebGL2 renderer for the emotional visualization system.
 *
 * Supports 18 shader demo modes (switchable at runtime) and
 * reaction-diffusion, smoke, erosion, and LBM simulations.
 */

import {
    createShader, createProgram, vertexShaderSource,
    fragmentShaderSources, rdSimulationFragmentSource,
    smokeSimulationFragmentSource, erosionSimulationFragmentSource,
    lbmSimulationFragmentSource,
    SHADER_MODE_NAMES,
} from './shaders.js';

/** Uniform locations cached per shader program */
interface ProgramUniforms {
    program: WebGLProgram;
    asciiMask: WebGLUniformLocation | null;
    emotionColors: WebGLUniformLocation | null;
    emotionValues: WebGLUniformLocation | null;
    emotionVelocities: WebGLUniformLocation | null;
    time: WebGLUniformLocation | null;
    prevFrame: WebGLUniformLocation | null;
    feedbackStrength: WebGLUniformLocation | null;
    enableFeedback: WebGLUniformLocation | null;
    // Mode 5 only
    rdState: WebGLUniformLocation | null;
}

/** R-D simulation program uniforms */
interface RDProgramUniforms {
    program: WebGLProgram;
    state: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
    time: WebGLUniformLocation | null;
    emotionValues: WebGLUniformLocation | null;
}

/** Generic simulation program uniforms (smoke, erosion, lbm) */
interface SimProgramUniforms {
    program: WebGLProgram;
    state: WebGLUniformLocation | null;
    resolution: WebGLUniformLocation | null;
    time: WebGLUniformLocation | null;
    emotionValues: WebGLUniformLocation | null;
    emotionVelocities: WebGLUniformLocation | null;
}

/** Simulation FBO pair */
interface SimResources {
    program: SimProgramUniforms;
    fbos: WebGLFramebuffer[];
    textures: WebGLTexture[];
    currentIndex: number;
    resolution: number;
    initialized: boolean;
    stepsPerFrame: number;
}

export class Renderer {
    private gl: WebGL2RenderingContext | null = null;
    private canvas: HTMLCanvasElement | null = null;

    // Geometry
    private vao: WebGLVertexArrayObject | null = null;
    private positionBuffer: WebGLBuffer | null = null;

    // Multi-program support
    private programs: ProgramUniforms[] = [];
    private currentMode: number = 0;

    // Current emotion state for feedback calculation
    private currentVelocities: number[] = [];

    // Reduced motion mode: disables temporal feedback
    private reducedMotion: boolean = false;

    // ASCII mask texture
    private asciiMaskTexture: WebGLTexture | null = null;

    // Ping-pong framebuffers for temporal feedback
    private fbos: WebGLFramebuffer[] = [];
    private fboTextures: WebGLTexture[] = [];
    private currentFboIndex: number = 0;

    // Reaction-diffusion resources (mode 5)
    private rdProgram: RDProgramUniforms | null = null;
    private rdFbos: WebGLFramebuffer[] = [];
    private rdTextures: WebGLTexture[] = [];
    private rdCurrentIndex: number = 0;
    private rdResolution: number = 256;
    private rdInitialized: boolean = false;
    private rdSimStepsPerFrame: number = 12;

    // Smoke simulation (mode 12)
    private smoke: SimResources | null = null;

    // Erosion simulation (mode 13)
    private erosion: SimResources | null = null;

    // LBM simulation (mode 17)
    private lbm: SimResources | null = null;

    // -----------------------------------------------------------------------
    // Initialization
    // -----------------------------------------------------------------------

    public init(canvas: HTMLCanvasElement): boolean {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');
        if (!this.gl) return false;

        try {
            this.initPrograms();
            this.initBuffers();
            this.createFramebuffers(canvas.width, canvas.height);
            this.initRDResources();
            this.initSimResource('smoke', smokeSimulationFragmentSource, 256, 4);
            this.initSimResource('erosion', erosionSimulationFragmentSource, 256, 2);
            this.initSimResource('lbm', lbmSimulationFragmentSource, 128, 4);
            return true;
        } catch (error) {
            console.error('Renderer init failed:', error);
            // Surface the real error in the fallback UI
            const fallback = document.getElementById('webgl-fallback');
            if (fallback) {
                const detail = document.createElement('pre');
                detail.style.cssText = 'margin-top:1em;font-size:0.75em;text-align:left;max-width:80ch;overflow:auto;color:#f88;';
                detail.textContent = String(error);
                fallback.appendChild(detail);
            }
            return false;
        }
    }

    private initPrograms(): void {
        const gl = this.gl!;
        const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);

        // Compile all 6 render programs
        for (let i = 0; i < fragmentShaderSources.length; i++) {
            const fragShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSources[i]);
            const program = createProgram(gl, vertexShader, fragShader);
            gl.deleteShader(fragShader);

            const pu: ProgramUniforms = {
                program,
                asciiMask: gl.getUniformLocation(program, 'u_asciiMask'),
                emotionColors: gl.getUniformLocation(program, 'u_emotionColors'),
                emotionValues: gl.getUniformLocation(program, 'u_emotionValues'),
                emotionVelocities: gl.getUniformLocation(program, 'u_emotionVelocities'),
                time: gl.getUniformLocation(program, 'u_time'),
                prevFrame: gl.getUniformLocation(program, 'uPrevFrame'),
                feedbackStrength: gl.getUniformLocation(program, 'uFeedbackStrength'),
                enableFeedback: gl.getUniformLocation(program, 'uEnableFeedback'),
                rdState: gl.getUniformLocation(program, 'uRDState'),
            };
            this.programs.push(pu);
        }

        gl.deleteShader(vertexShader);
    }

    private initRDResources(): void {
        const gl = this.gl!;

        // Compile R-D simulation program
        const vtx = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
        const frag = createShader(gl, gl.FRAGMENT_SHADER, rdSimulationFragmentSource);
        const prog = createProgram(gl, vtx, frag);
        gl.deleteShader(vtx);
        gl.deleteShader(frag);

        this.rdProgram = {
            program: prog,
            state: gl.getUniformLocation(prog, 'uState'),
            resolution: gl.getUniformLocation(prog, 'uResolution'),
            time: gl.getUniformLocation(prog, 'u_time'),
            emotionValues: gl.getUniformLocation(prog, 'u_emotionValues'),
        };

        // Create R-D ping-pong FBOs at half resolution
        this.createRDFramebuffers();
    }

    private createRDFramebuffers(): void {
        const gl = this.gl!;
        const res = this.rdResolution;

        // Check for float texture support
        const ext = gl.getExtension('EXT_color_buffer_float');
        const internalFormat = ext ? gl.RGBA32F : gl.RGBA;
        const type = ext ? gl.FLOAT : gl.UNSIGNED_BYTE;

        for (let i = 0; i < 2; i++) {
            const fbo = gl.createFramebuffer()!;
            const tex = gl.createTexture()!;
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat as number, res, res, 0, gl.RGBA, type, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
            this.rdFbos.push(fbo);
            this.rdTextures.push(tex);
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        this.rdInitialized = false;
    }

    /** Fill R-D state with U=1, V=0 */
    private initRDState(): void {
        const gl = this.gl!;
        const res = this.rdResolution;

        // Check if we have float textures
        const ext = gl.getExtension('EXT_color_buffer_float');

        if (ext) {
            const data = new Float32Array(res * res * 4);
            for (let i = 0; i < res * res; i++) {
                data[i * 4 + 0] = 1.0; // U
                data[i * 4 + 1] = 0.0; // V
                data[i * 4 + 2] = 0.0;
                data[i * 4 + 3] = 1.0;
            }
            for (let t = 0; t < 2; t++) {
                gl.bindTexture(gl.TEXTURE_2D, this.rdTextures[t]);
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, res, res, gl.RGBA, gl.FLOAT, data);
            }
        } else {
            const data = new Uint8Array(res * res * 4);
            for (let i = 0; i < res * res; i++) {
                data[i * 4 + 0] = 255; // U = 1.0
                data[i * 4 + 1] = 0;   // V = 0.0
                data[i * 4 + 2] = 0;
                data[i * 4 + 3] = 255;
            }
            for (let t = 0; t < 2; t++) {
                gl.bindTexture(gl.TEXTURE_2D, this.rdTextures[t]);
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, res, res, gl.RGBA, gl.UNSIGNED_BYTE, data);
            }
        }

        gl.bindTexture(gl.TEXTURE_2D, null);
        this.rdCurrentIndex = 0;
        this.rdInitialized = true;
    }

    // -----------------------------------------------------------------------
    // Buffers
    // -----------------------------------------------------------------------

    private initBuffers(): void {
        const gl = this.gl!;
        this.vao = gl.createVertexArray()!;
        gl.bindVertexArray(this.vao);
        this.positionBuffer = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1
        ]), gl.STATIC_DRAW);

        const posLoc = gl.getAttribLocation(this.programs[0].program, 'a_position');
        gl.enableVertexAttribArray(posLoc);
        gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
    }

    // -----------------------------------------------------------------------
    // Framebuffers (temporal feedback)
    // -----------------------------------------------------------------------

    private createFramebuffers(width: number, height: number): void {
        const gl = this.gl!;
        for (let i = 0; i < 2; i++) {
            const fbo = gl.createFramebuffer()!;
            const tex = gl.createTexture()!;
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
            this.fbos.push(fbo);
            this.fboTextures.push(tex);
        }
        // Clear both
        for (let i = 0; i < 2; i++) {
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[i]);
            gl.clearColor(0, 0, 0, 1);
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    private deleteFramebuffers(): void {
        const gl = this.gl!;
        for (const fbo of this.fbos) gl.deleteFramebuffer(fbo);
        for (const tex of this.fboTextures) gl.deleteTexture(tex);
        this.fbos = [];
        this.fboTextures = [];
        this.currentFboIndex = 0;
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    public resize(width: number, height: number): void {
        if (!this.canvas || !this.gl) return;
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl.viewport(0, 0, width, height);
        this.deleteFramebuffers();
        this.createFramebuffers(width, height);
    }

    public setShaderMode(mode: number): void {
        if (mode < 0 || mode >= this.programs.length) return;
        this.currentMode = mode;
        if (mode === 5) this.rdInitialized = false;
        if (mode === 12 && this.smoke) this.smoke.initialized = false;
        if (mode === 13 && this.erosion) this.erosion.initialized = false;
        if (mode === 17 && this.lbm) this.lbm.initialized = false;
    }

    public getShaderMode(): number {
        return this.currentMode;
    }

    public setColor(r: number, g: number, b: number): void {
        // Legacy — no longer used
    }

    public setEmotionData(colors: [number, number, number][], values: number[], velocities: number[]): void {
        if (!this.gl) return;
        const gl = this.gl;
        // Set uniforms on ALL programs so switching modes mid-frame is safe
        const flatColors: number[] = [];
        for (const c of colors) flatColors.push(c[0] / 255, c[1] / 255, c[2] / 255);

        for (const pu of this.programs) {
            gl.useProgram(pu.program);
            gl.uniform3fv(pu.emotionColors, flatColors);
            gl.uniform1fv(pu.emotionValues, values);
            gl.uniform1fv(pu.emotionVelocities, velocities);
        }
        // Also set on R-D simulation program
        if (this.rdProgram) {
            gl.useProgram(this.rdProgram.program);
            gl.uniform1fv(this.rdProgram.emotionValues, values);
        }
        // Set on simulation programs
        for (const sim of [this.smoke, this.erosion, this.lbm]) {
            if (sim) {
                gl.useProgram(sim.program.program);
                gl.uniform1fv(sim.program.emotionValues, values);
                gl.uniform1fv(sim.program.emotionVelocities, velocities);
            }
        }
        this.currentVelocities = [...velocities];
    }

    public setTime(time: number): void {
        if (!this.gl) return;
        const gl = this.gl;
        for (const pu of this.programs) {
            gl.useProgram(pu.program);
            gl.uniform1f(pu.time, time);
        }
        if (this.rdProgram) {
            gl.useProgram(this.rdProgram.program);
            gl.uniform1f(this.rdProgram.time, time);
        }
        for (const sim of [this.smoke, this.erosion, this.lbm]) {
            if (sim) {
                gl.useProgram(sim.program.program);
                gl.uniform1f(sim.program.time, time);
            }
        }
    }

    public setASCIIMask(texture: WebGLTexture): void {
        this.asciiMaskTexture = texture;
    }

    /**
     * Set reduced motion mode. When enabled, temporal feedback is disabled
     * and noise distortion is reduced for users who prefer reduced motion.
     */
    public setReducedMotion(enabled: boolean): void {
        this.reducedMotion = enabled;
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    public render(): void {
        if (!this.gl || this.programs.length === 0 || !this.vao || this.fbos.length < 2) return;
        const gl = this.gl;

        // Run R-D simulation if in mode 5
        if (this.currentMode === 5) {
            this.runRDSimulation();
        }

        // Run smoke simulation if in mode 12
        if (this.currentMode === 12 && this.smoke) {
            this.runSimStep(this.smoke);
        }

        // Run erosion simulation if in mode 13
        if (this.currentMode === 13 && this.erosion) {
            this.runSimStep(this.erosion);
        }

        // Run LBM simulation if in mode 17
        if (this.currentMode === 17 && this.lbm) {
            this.runSimStep(this.lbm);
        }

        const pu = this.programs[this.currentMode];
        const prevFboIndex = (this.currentFboIndex + 1) % 2;

        gl.useProgram(pu.program);

        // TEXTURE0: ASCII mask
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.asciiMaskTexture);
        gl.uniform1i(pu.asciiMask, 0);

        // TEXTURE1: Previous frame
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.fboTextures[prevFboIndex]);
        gl.uniform1i(pu.prevFrame, 1);

        // TEXTURE2: Simulation state
        if (this.currentMode === 5 && pu.rdState !== null) {
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, this.rdTextures[this.rdCurrentIndex]);
            gl.uniform1i(pu.rdState, 2);
        }
        if (this.currentMode === 12 && this.smoke) {
            const loc = gl.getUniformLocation(pu.program, 'uSmokeState');
            if (loc !== null) {
                gl.activeTexture(gl.TEXTURE2);
                gl.bindTexture(gl.TEXTURE_2D, this.smoke.textures[this.smoke.currentIndex]);
                gl.uniform1i(loc, 2);
            }
        }
        if (this.currentMode === 13 && this.erosion) {
            const loc = gl.getUniformLocation(pu.program, 'uErosionState');
            if (loc !== null) {
                gl.activeTexture(gl.TEXTURE2);
                gl.bindTexture(gl.TEXTURE_2D, this.erosion.textures[this.erosion.currentIndex]);
                gl.uniform1i(loc, 2);
            }
        }
        if (this.currentMode === 17 && this.lbm) {
            const loc = gl.getUniformLocation(pu.program, 'uLBMState');
            if (loc !== null) {
                gl.activeTexture(gl.TEXTURE2);
                gl.bindTexture(gl.TEXTURE_2D, this.lbm.textures[this.lbm.currentIndex]);
                gl.uniform1i(loc, 2);
            }
        }

        // Feedback strength from velocity (disabled in reduced motion mode)
        if (this.reducedMotion) {
            gl.uniform1f(pu.feedbackStrength, 0.0);
            gl.uniform1f(pu.enableFeedback, 0.0);
        } else {
            let totalVelocity = 0;
            for (const v of this.currentVelocities) totalVelocity += v;
            const feedbackStrength = Math.min(totalVelocity * 0.5, 0.95);
            gl.uniform1f(pu.feedbackStrength, Math.max(0, Math.min(feedbackStrength, 0.95)));
            gl.uniform1f(pu.enableFeedback, 1.0);
        }

        gl.bindVertexArray(this.vao);

        // Pass 1: Render to current FBO
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbos[this.currentFboIndex]);
        gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // Swap FBO indices
        this.currentFboIndex = prevFboIndex;

        // Pass 2: Display to screen
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);
        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.uniform1f(pu.enableFeedback, 0.0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.fboTextures[this.currentFboIndex]);
        gl.drawArrays(gl.TRIANGLES, 0, 6);

        gl.bindVertexArray(null);
    }

    // -----------------------------------------------------------------------
    // Reaction-Diffusion simulation
    // -----------------------------------------------------------------------

    private runRDSimulation(): void {
        if (!this.rdProgram || this.rdFbos.length < 2) return;
        const gl = this.gl!;

        // Initialize state on first use
        if (!this.rdInitialized) this.initRDState();

        gl.useProgram(this.rdProgram.program);
        gl.uniform2f(this.rdProgram.resolution, this.rdResolution, this.rdResolution);
        gl.bindVertexArray(this.vao);

        for (let step = 0; step < this.rdSimStepsPerFrame; step++) {
            const readIdx = this.rdCurrentIndex;
            const writeIdx = (this.rdCurrentIndex + 1) % 2;

            // Bind read texture
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, this.rdTextures[readIdx]);
            gl.uniform1i(this.rdProgram.state, 0);

            // Render to write FBO
            gl.bindFramebuffer(gl.FRAMEBUFFER, this.rdFbos[writeIdx]);
            gl.viewport(0, 0, this.rdResolution, this.rdResolution);
            gl.drawArrays(gl.TRIANGLES, 0, 6);

            this.rdCurrentIndex = writeIdx;
        }

        // Restore viewport
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);
        gl.bindVertexArray(null);
    }

    private initSimResource(
        target: 'smoke' | 'erosion' | 'lbm',
        fragSource: string,
        resolution: number,
        stepsPerFrame: number,
    ): void {
        const gl = this.gl!;
        const vtx = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
        const frag = createShader(gl, gl.FRAGMENT_SHADER, fragSource);
        const prog = createProgram(gl, vtx, frag);
        gl.deleteShader(vtx);
        gl.deleteShader(frag);

        const uniforms: SimProgramUniforms = {
            program: prog,
            state: gl.getUniformLocation(prog, 'uState'),
            resolution: gl.getUniformLocation(prog, 'uResolution'),
            time: gl.getUniformLocation(prog, 'u_time'),
            emotionValues: gl.getUniformLocation(prog, 'u_emotionValues'),
            emotionVelocities: gl.getUniformLocation(prog, 'u_emotionVelocities'),
        };

        const fbos: WebGLFramebuffer[] = [];
        const textures: WebGLTexture[] = [];

        for (let i = 0; i < 2; i++) {
            const fbo = gl.createFramebuffer()!;
            const tex = gl.createTexture()!;
            gl.bindTexture(gl.TEXTURE_2D, tex);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, resolution, resolution, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
            gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
            fbos.push(fbo);
            textures.push(tex);
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);

        this[target] = {
            program: uniforms,
            fbos,
            textures,
            currentIndex: 0,
            resolution,
            initialized: false,
            stepsPerFrame,
        };
    }

    private initSimState(sim: SimResources): void {
        const gl = this.gl!;
        const res = sim.resolution;
        const data = new Uint8Array(res * res * 4);
        for (let i = 0; i < res * res; i++) {
            data[i * 4 + 3] = 255;
        }
        for (let t = 0; t < 2; t++) {
            gl.bindTexture(gl.TEXTURE_2D, sim.textures[t]);
            gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, res, res, gl.RGBA, gl.UNSIGNED_BYTE, data);
        }
        gl.bindTexture(gl.TEXTURE_2D, null);
        sim.currentIndex = 0;
        sim.initialized = true;
    }

    private runSimStep(sim: SimResources): void {
        if (!sim || sim.fbos.length < 2) return;
        const gl = this.gl!;

        if (!sim.initialized) this.initSimState(sim);

        gl.useProgram(sim.program.program);
        gl.uniform2f(sim.program.resolution, sim.resolution, sim.resolution);
        gl.bindVertexArray(this.vao);

        for (let step = 0; step < sim.stepsPerFrame; step++) {
            const readIdx = sim.currentIndex;
            const writeIdx = (sim.currentIndex + 1) % 2;

            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, sim.textures[readIdx]);
            gl.uniform1i(sim.program.state, 0);

            gl.bindFramebuffer(gl.FRAMEBUFFER, sim.fbos[writeIdx]);
            gl.viewport(0, 0, sim.resolution, sim.resolution);
            gl.drawArrays(gl.TRIANGLES, 0, 6);

            sim.currentIndex = writeIdx;
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas!.width, this.canvas!.height);
        gl.bindVertexArray(null);
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    public isWebGL2Available(): boolean {
        return this.gl !== null;
    }

    public dispose(): void {
        if (!this.gl) return;
        const gl = this.gl;
        this.deleteFramebuffers();
        for (const fbo of this.rdFbos) gl.deleteFramebuffer(fbo);
        for (const tex of this.rdTextures) gl.deleteTexture(tex);
        for (const sim of [this.smoke, this.erosion, this.lbm]) {
            if (sim) {
                for (const fbo of sim.fbos) gl.deleteFramebuffer(fbo);
                for (const tex of sim.textures) gl.deleteTexture(tex);
                gl.deleteProgram(sim.program.program);
            }
        }
        if (this.positionBuffer) gl.deleteBuffer(this.positionBuffer);
        if (this.vao) gl.deleteVertexArray(this.vao);
        for (const pu of this.programs) gl.deleteProgram(pu.program);
        if (this.rdProgram) gl.deleteProgram(this.rdProgram.program);
    }
}
