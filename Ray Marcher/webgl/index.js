function createShader(gl, type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
    if (success) {
        return shader;
    }
   
    console.log(gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
}

function createProgram(gl, vertexShader, fragmentShader) {
    var program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    var success = gl.getProgramParameter(program, gl.LINK_STATUS);
    if (success) {
        return program;
    }
   
    console.log(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
}

window.onload = () => {
    let canvas = document.querySelector("canvas");
    let gl = canvas.getContext("webgl2");

    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
 
    var vertexShaderSource =
    `#version 300 es
    in vec4 a_position;

    void main() {
        gl_Position = a_position;
    }`;
    var fragmentShaderSource =
    `#version 300 es
    precision mediump float;

    out vec4 outColor; 
    void main() {
        outColor = vec4(gl_FragCoord.x / 640.0, gl_FragCoord.y / 480.0, 0, 1);
    }`;
    let buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
            -1,  1,
             1, -1,
             1,  1
        ]),
        gl.STATIC_DRAW
    );

    let vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    let fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    let program = createProgram(gl, vertexShader, fragmentShader);

    gl.useProgram(program);

    let positionLocation = gl.getAttribLocation(program, "a_position");
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);


    
    function update() {
        
        window.requestAnimationFrame(update);
    }
    //update();
}