/**
 * ASCII mask texture generation using Canvas2D preprocessing and WebGL2.
 * 
 * This module handles the conversion of ASCII art into GPU textures by:
 * 1. Rendering ASCII glyphs to an offscreen Canvas2D
 * 2. Uploading the canvas as an alpha texture to WebGL
 * 3. Providing resolution-independent texture sampling
 */

/**
 * ASCIIMask class for generating and managing ASCII art textures
 */
export class ASCIIMask {
    private gl: WebGL2RenderingContext | null = null;
    private texture: WebGLTexture | null = null;
    private canvas: HTMLCanvasElement | null = null;
    private ctx: CanvasRenderingContext2D | null = null;
    
    // Font settings
    private fontSize: number = 16;
    private fontFamily: string = "'IBM Plex Mono', monospace";
    
    /**
     * Initialize the ASCII mask with WebGL2 context
     * @param gl - The WebGL2 rendering context
     */
    public init(gl: WebGL2RenderingContext): void {
        this.gl = gl;
    }
    
    /**
     * Generate a texture mask from ASCII art string
     * @param asciiArt - Multi-line string containing ASCII art
     * @param width - Width in pixels for the texture
     * @param height - Height in pixels for the texture
     * @returns The generated WebGL texture
     */
    public generateMask(asciiArt: string, width: number, height: number): WebGLTexture | null {
        if (!this.gl) {
            return null;
        }
        
        // Parse ASCII art to get dimensions
        const lines = asciiArt.split('\n');
        const charWidth = Math.max(...lines.map(line => line.length));
        const charHeight = lines.length;
        
        if (charWidth === 0 || charHeight === 0) {
            return null;
        }
        
        // Calculate canvas size based on character grid
        const canvasWidth = width;
        const canvasHeight = height;
        
        // Create offscreen canvas for rendering
        this.canvas = document.createElement('canvas');
        this.canvas.width = canvasWidth;
        this.canvas.height = canvasHeight;
        
        const ctx = this.canvas.getContext('2d');
        if (!ctx) {
            return null;
        }
        this.ctx = ctx;
        
        // Clear canvas with transparent background
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
        // Set font properties
        const scaledFontSize = Math.floor(canvasHeight / charHeight);
        ctx.font = `${scaledFontSize}px ${this.fontFamily}`;
        ctx.textBaseline = 'top';
        ctx.fillStyle = 'white';
        
        // Calculate character dimensions based on font metrics
        const charPixelWidth = canvasWidth / charWidth;
        const charPixelHeight = canvasHeight / charHeight;
        
        // Measure actual character width to center it
        const metrics = ctx.measureText('M');
        const actualCharWidth = metrics.width;
        
        // Render each character
        for (let y = 0; y < charHeight; y++) {
            const line = lines[y] || '';
            for (let x = 0; x < line.length; x++) {
                const char = line[x];
                
                // Skip space characters (leave transparent)
                if (char === ' ') {
                    continue;
                }
                
                // Calculate position - use integer coordinates to avoid pixel snapping
                const posX = Math.floor(x * charPixelWidth);
                const posY = Math.floor(y * charPixelHeight);
                
                // Center the character within its cell
                const offsetX = Math.floor((charPixelWidth - actualCharWidth) / 2);
                const offsetY = Math.floor((charPixelHeight - scaledFontSize) / 2);
                
                // Draw the character as white on transparent background
                ctx.fillText(char, posX + offsetX, posY + offsetY);
            }
        }
        
        // Create and upload texture to WebGL
        this.texture = this.uploadTexture();
        
        return this.texture;
    }
    
    /**
     * Upload the canvas content as a WebGL texture
     * @returns The created WebGL texture
     */
    private uploadTexture(): WebGLTexture | null {
        if (!this.gl || !this.canvas) {
            return null;
        }
        
        const gl = this.gl;
        
        // Create texture
        const texture = gl.createTexture();
        if (!texture) {
            return null;
        }
        
        // Bind texture
        gl.bindTexture(gl.TEXTURE_2D, texture);
        
        // Upload canvas data as alpha texture
        // Using RGBA format with white glyphs on transparent background
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,                    // mip level
            gl.RGBA,              // internal format
            gl.RGBA,              // source format
            gl.UNSIGNED_BYTE,     // source type
            this.canvas           // source (canvas element)
        );
        
        // Configure texture parameters for linear filtering
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        
        // Generate mipmaps for better quality at different scales
        gl.generateMipmap(gl.TEXTURE_2D);
        
        return texture;
    }
    
    /**
     * Get the generated WebGL texture
     * @returns The WebGL texture or null if not generated
     */
    public getTexture(): WebGLTexture | null {
        return this.texture;
    }
    
    /**
     * Clean up texture and canvas resources
     */
    public dispose(): void {
        // Delete WebGL texture
        if (this.gl && this.texture) {
            this.gl.deleteTexture(this.texture);
            this.texture = null;
        }
        
        // Canvas will be garbage collected
        this.canvas = null;
        this.ctx = null;
    }
}
