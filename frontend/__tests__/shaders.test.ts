import { describe, it, expect } from 'vitest';
import {
    SHADER_MODE_NAMES,
    fragmentShaderSources,
    rdSimulationFragmentSource,
} from '../gl/shaders.js';

// ---------------------------------------------------------------------------
// Registry integrity
// ---------------------------------------------------------------------------

describe('Shader mode registry', () => {
    it('has 18 registered modes', () => {
        expect(SHADER_MODE_NAMES.length).toBe(18);
    });

    it('has matching number of fragment shader sources', () => {
        expect(fragmentShaderSources.length).toBe(SHADER_MODE_NAMES.length);
    });

    it('has no empty mode names', () => {
        for (const name of SHADER_MODE_NAMES) {
            expect(name.length).toBeGreaterThan(0);
        }
    });

    it('has no empty shader sources', () => {
        for (const src of fragmentShaderSources) {
            expect(src.length).toBeGreaterThan(0);
        }
    });

    it('has unique mode names', () => {
        const unique = new Set(SHADER_MODE_NAMES);
        expect(unique.size).toBe(SHADER_MODE_NAMES.length);
    });

    it('has enough key bindings for all modes (18 = 10 digits + 8 letters)', () => {
        expect(SHADER_MODE_NAMES.length).toBeLessThanOrEqual(18);
    });
});

// ---------------------------------------------------------------------------
// Common shader contract tests
// ---------------------------------------------------------------------------

function shaderContractTests(modeIndex: number, modeName: string, options?: {
    requiresPrevFrame?: boolean;
    requiresSimState?: string; // e.g. 'uRDState', 'uSmokeState'
}) {
    describe(`Mode ${modeIndex}: ${modeName}`, () => {
        it(`is registered as "${modeName}"`, () => {
            expect(SHADER_MODE_NAMES[modeIndex]).toBe(modeName);
        });

        it('has a corresponding fragment shader source', () => {
            expect(fragmentShaderSources[modeIndex]).toBeDefined();
            expect(fragmentShaderSources[modeIndex].length).toBeGreaterThan(0);
        });

        it('contains void main function', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('void main()');
        });

        it('outputs fragColor', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('fragColor');
        });

        it('references emotion values', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('u_emotionValues');
        });

        it('references emotion velocities', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('u_emotionVelocities');
        });

        it('uses OKLab blending', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('srgbToOkLab');
        });

        it('applies ASCII mask', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('u_asciiMask');
        });

        it('uses #version 300 es', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('#version 300 es');
        });

        it('uses highp precision', () => {
            expect(fragmentShaderSources[modeIndex]).toContain('precision highp float');
        });

        if (options?.requiresPrevFrame) {
            it('references prevFrame texture', () => {
                expect(fragmentShaderSources[modeIndex]).toContain('uPrevFrame');
            });
        }

        if (options?.requiresSimState) {
            it(`references simulation state uniform "${options.requiresSimState}"`, () => {
                expect(fragmentShaderSources[modeIndex]).toContain(options.requiresSimState!);
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Existing modes (0-5) — sanity checks
// ---------------------------------------------------------------------------

describe('Existing modes sanity check', () => {
    it('Mode 0 is Voronoi', () => {
        expect(SHADER_MODE_NAMES[0]).toBe('Voronoi');
    });
    it('Mode 5 is Reaction-Diffusion', () => {
        expect(SHADER_MODE_NAMES[5]).toBe('Reaction-Diffusion');
    });
    it('Mode 5 render shader references uRDState', () => {
        expect(fragmentShaderSources[5]).toContain('uRDState');
    });
});

// ---------------------------------------------------------------------------
// Mode 6: Chladni Resonance
// ---------------------------------------------------------------------------

shaderContractTests(6, 'Chladni');

describe('Mode 6: Chladni specifics', () => {
    it('uses mode pair calculations for standing waves', () => {
        expect(fragmentShaderSources[6]).toContain('cos');
    });
    it('references totalVel for nodal line width', () => {
        expect(fragmentShaderSources[6]).toContain('totalVel');
    });
    it('applies motion vector feedback', () => {
        expect(fragmentShaderSources[6]).toContain('applyMotionVectorFeedback');
    });
});

// ---------------------------------------------------------------------------
// Mode 7: Cymatics / Wave Interference
// ---------------------------------------------------------------------------

shaderContractTests(7, 'Cymatics');

describe('Mode 7: Cymatics specifics', () => {
    it('uses sin for wave interference', () => {
        expect(fragmentShaderSources[7]).toContain('sin(');
    });
    it('uses distance calculations for point sources', () => {
        expect(fragmentShaderSources[7]).toContain('length(');
    });
});

// ---------------------------------------------------------------------------
// Mode 8: Julia Set
// ---------------------------------------------------------------------------

shaderContractTests(8, 'Julia Set');

describe('Mode 8: Julia Set specifics', () => {
    it('contains iteration loop for fractal computation', () => {
        expect(fragmentShaderSources[8]).toContain('for');
    });
    it('uses early bailout', () => {
        expect(fragmentShaderSources[8]).toContain('4.0');
    });
    it('applies motion vector feedback', () => {
        expect(fragmentShaderSources[8]).toContain('applyMotionVectorFeedback');
    });
});

// ---------------------------------------------------------------------------
// Mode 9: Gravitational Lensing
// ---------------------------------------------------------------------------

shaderContractTests(9, 'Gravity Lens', { requiresPrevFrame: true });

describe('Mode 9: Gravity Lens specifics', () => {
    it('uses velocities for frame-dragging', () => {
        expect(fragmentShaderSources[9]).toContain('u_emotionVelocities');
    });
    it('clamps Jacobian determinant', () => {
        expect(fragmentShaderSources[9]).toContain('max(');
    });
});

// ---------------------------------------------------------------------------
// Mode 10: Strange Attractor (Clifford)
// ---------------------------------------------------------------------------

shaderContractTests(10, 'Attractor');

describe('Mode 10: Attractor specifics', () => {
    it('uses trigonometric functions for attractor computation', () => {
        expect(fragmentShaderSources[10]).toContain('sin(');
        expect(fragmentShaderSources[10]).toContain('cos(');
    });
    it('references u_time for temporal drift', () => {
        expect(fragmentShaderSources[10]).toContain('u_time');
    });
});

// ---------------------------------------------------------------------------
// Mode 11: Voronoi Cracks
// ---------------------------------------------------------------------------

shaderContractTests(11, 'Cracks');

describe('Mode 11: Cracks specifics', () => {
    it('uses velocities for crack width', () => {
        expect(fragmentShaderSources[11]).toContain('u_emotionVelocities');
    });
    it('uses noise for crack edge perturbation', () => {
        expect(fragmentShaderSources[11]).toContain('snoise(');
    });
    it('applies motion vector feedback', () => {
        expect(fragmentShaderSources[11]).toContain('applyMotionVectorFeedback');
    });
});

// ---------------------------------------------------------------------------
// Mode 12: Smoke
// ---------------------------------------------------------------------------

shaderContractTests(12, 'Smoke', { requiresSimState: 'uSmokeState' });

describe('Mode 12: Smoke specifics', () => {
    it('has a simulation fragment shader', () => {
        // smokeSimulationFragmentSource will be tested via named export
        expect(fragmentShaderSources[12]).toContain('uSmokeState');
    });
});

// ---------------------------------------------------------------------------
// Mode 13: Topography / Erosion
// ---------------------------------------------------------------------------

shaderContractTests(13, 'Topography', { requiresSimState: 'uErosionState' });

describe('Mode 13: Topography specifics', () => {
    it('has a simulation fragment shader', () => {
        expect(fragmentShaderSources[13]).toContain('uErosionState');
    });
});

// ---------------------------------------------------------------------------
// Mode 14: Magnetic LIC
// ---------------------------------------------------------------------------

shaderContractTests(14, 'Magnetic LIC', { requiresPrevFrame: true });

describe('Mode 14: Magnetic LIC specifics', () => {
    it('uses temporal feedback for progressive LIC', () => {
        expect(fragmentShaderSources[14]).toContain('uPrevFrame');
    });
});

// ---------------------------------------------------------------------------
// Mode 15: Lissajous
// ---------------------------------------------------------------------------

shaderContractTests(15, 'Lissajous', { requiresPrevFrame: true });

describe('Mode 15: Lissajous specifics', () => {
    it('uses parametric sin/cos for curves', () => {
        expect(fragmentShaderSources[15]).toContain('sin(');
        expect(fragmentShaderSources[15]).toContain('cos(');
    });
});

// ---------------------------------------------------------------------------
// Mode 16: Phyllotaxis
// ---------------------------------------------------------------------------

shaderContractTests(16, 'Phyllotaxis');

describe('Mode 16: Phyllotaxis specifics', () => {
    it('uses golden angle calculation', () => {
        expect(fragmentShaderSources[16]).toContain('2.399963'); // golden angle in radians
    });
    it('applies motion vector feedback', () => {
        expect(fragmentShaderSources[16]).toContain('applyMotionVectorFeedback');
    });
});

// ---------------------------------------------------------------------------
// Mode 17: Lattice Boltzmann
// ---------------------------------------------------------------------------

shaderContractTests(17, 'Ink Flow', { requiresSimState: 'uLBMState' });

describe('Mode 17: Ink Flow specifics', () => {
    it('references simulation state', () => {
        expect(fragmentShaderSources[17]).toContain('uLBMState');
    });
});

// ---------------------------------------------------------------------------
// Simulation shader exports
// ---------------------------------------------------------------------------

describe('Simulation shader exports', () => {
    it('exports rdSimulationFragmentSource', () => {
        expect(rdSimulationFragmentSource).toBeDefined();
        expect(rdSimulationFragmentSource.length).toBeGreaterThan(0);
    });

    it('rdSimulationFragmentSource contains uState and uResolution', () => {
        expect(rdSimulationFragmentSource).toContain('uState');
        expect(rdSimulationFragmentSource).toContain('uResolution');
    });
});

// ---------------------------------------------------------------------------
// Smoke simulation shader
// ---------------------------------------------------------------------------

describe('Smoke simulation shader', () => {
    it('exports smokeSimulationFragmentSource', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.smokeSimulationFragmentSource).toBeDefined();
        expect(mod.smokeSimulationFragmentSource.length).toBeGreaterThan(0);
    });

    it('contains uState and uResolution', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.smokeSimulationFragmentSource).toContain('uState');
        expect(mod.smokeSimulationFragmentSource).toContain('uResolution');
    });

    it('clamps state channels', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.smokeSimulationFragmentSource).toContain('clamp(');
    });
});

// ---------------------------------------------------------------------------
// Erosion simulation shader
// ---------------------------------------------------------------------------

describe('Erosion simulation shader', () => {
    it('exports erosionSimulationFragmentSource', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.erosionSimulationFragmentSource).toBeDefined();
        expect(mod.erosionSimulationFragmentSource.length).toBeGreaterThan(0);
    });

    it('contains uState and uResolution', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.erosionSimulationFragmentSource).toContain('uState');
        expect(mod.erosionSimulationFragmentSource).toContain('uResolution');
    });
});

// ---------------------------------------------------------------------------
// LBM simulation shader
// ---------------------------------------------------------------------------

describe('LBM simulation shader', () => {
    it('exports lbmSimulationFragmentSource', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.lbmSimulationFragmentSource).toBeDefined();
        expect(mod.lbmSimulationFragmentSource.length).toBeGreaterThan(0);
    });

    it('contains uResolution', async () => {
        const mod = await import('../gl/shaders.js');
        expect(mod.lbmSimulationFragmentSource).toContain('uResolution');
    });
});
