let __vertexShaderSource;
let __fragmentShaderSource
let __objects;
let __lights;
let __samples = 64;
let __raySteps = 4;
let camera;
let canvas;

class Vector {
    constructor(arr) {
        this.arr = arr;
    }
    get(i) {
        return this.arr[i];
    }
    set(i) {
        return this.arr[i];
    }
    add(x) {
        let newArr = [];
        if ("number" == typeof x) {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] + x);
            }
        } else {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] + x.arr[i]);
            }
        }
        return new Vector(newArr);
    }
    subtract(x) {
        let newArr = [];
        if ("number" == typeof x) {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] - x);
            }
        } else {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] - x.arr[i]);
            }
        }
        return new Vector(newArr);
    }
    multiply(x) {
        let newArr = [];
        if ("number" == typeof x) {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] * x);
            }
        } else {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] * x.arr[i]);
            }
        }
        return new Vector(newArr);
    }
    pow(x) {
        let newArr = [];
        if ("number" == typeof x) {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] ** x);
            }
        } else {
            for (let i = 0; i < this.arr.length; i++) {
                newArr.push(this.arr[i] ** x.arr[i]);
            }
        }
        return new Vector(newArr);
    }
    get sum() {
        let sum = 0;
        for (let i = 0; i < this.arr.length; i++) {
            sum += this.arr[i];
        }
        return sum;
    }
    get length() {
        let sum = 0;
        for (let i = 0; i < this.arr.length; i++) {
            sum += this.arr[i] ** 2;
        }
        return Math.sqrt(sum);
    }
    get normalized() {
        let newArr = [];
        let length = this.length;
        for (let i = 0; i < this.arr.length; i++) {
            newArr.push(this.arr[i] / length);
        }
        return new Vector(newArr);
    }
    rotateX(angle) {
        let newArr = [];
        angle = angle * Math.PI / 180;
        let sinX = Math.sin(angle);
        let cosX = Math.cos(angle);
        newArr.push(this.arr[0]);
        newArr.push(this.arr[1] * cosX - this.arr[2] * sinX);
        newArr.push(this.arr[1] * sinX + this.arr[2] * cosX);
        return new Vector(newArr);
    }
    rotateY(angle) {
        let newArr = [];
        angle = angle * Math.PI / 180;
        let sinY = Math.sin(angle);
        let cosY = Math.cos(angle);
        newArr.push(this.arr[0] * cosY + this.arr[2] * sinY);
        newArr.push(this.arr[1]);
        newArr.push(-this.arr[0] * sinY + this.arr[2] * cosY);
        return new Vector(newArr);
    }
    rotateZ(angle) {
        let newArr = [];
        angle = angle * Math.PI / 180;
        let sinZ = Math.sin(angle);
        let cosZ = Math.cos(angle);
        newArr.push(this.arr[0] * cosZ - this.arr[1] * sinZ);
        newArr.push(this.arr[0] * sinZ + this.arr[1] * cosZ);
        newArr.push(this.arr[2]);
        return new Vector(newArr);
    }
    show() {
        console.table(this.arr)
    }
}

class Material {
    constructor(color, albedo) {
        this.color = color;
        this.albedo = albedo;
    }
}

class Sphere {
    constructor(pos, material, radius) {
        this.pos = pos;
        this.radius = radius;
        this.material = material;
        __objects.addObject(this, "sphere");
    }
}

class Plane {
    constructor(normal, material) {
        this.normal = normal;
        this.material = material;
        __objects.addObject(this, "plane");
    }
}

class Light {
    constructor(pos, strength, color) {
        this.pos = pos;
        this.strength = strength;
        this.color = color;
        __lights.addLight(this);
    }
}

class Objects3D {
    constructor(gl, prg) {
        this.gl = gl;
        this.prg = prg;
        this.objects = [];
    }
    addObject(object, type) {
        this.objects.push({
            "object": object,
            "type": type
        });
    }
    update() {
        for (let i = 0; i < this.objects.length; i++) {
            let activeLoc = this.gl.getUniformLocation(this.prg, "activeObjects[" + i + "]");
            this.gl.uniform1i(activeLoc, 1);
            if (this.objects[i].type == "sphere") {
                let sphereLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].sphere");
                this.gl.uniform1i(sphereLoc, 1);
                let posLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].pos");
                this.gl.uniform3fv(posLoc, this.objects[i].object.pos.arr);
                let radiusLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].arg1");
                this.gl.uniform1f(radiusLoc, this.objects[i].object.radius);
            }
            if (this.objects[i].type == "plane") {
                let sphereLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].plane");
                this.gl.uniform1i(sphereLoc, 1);
                let normalLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].pos");
                this.gl.uniform3fv(normalLoc, this.objects[i].object.normal.arr);
            }
            let colorLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].material.color");
            this.gl.uniform3fv(colorLoc, this.objects[i].object.material.color.arr);
            let diffuseLoc = this.gl.getUniformLocation(this.prg, "objects[" + i + "].material.albedo");
            this.gl.uniform1f(diffuseLoc, this.objects[i].object.material.albedo);
        }
    }
}

class Lights {
    constructor(gl, prg) {
        this.gl = gl;
        this.prg = prg;
        this.lights = [];
    }
    addLight(light) {
        this.lights.push(light);
    }
    update() {
        for (let i = 0; i < this.lights.length; i++) {
            let activeLoc = this.gl.getUniformLocation(this.prg, "activeLights[" + i + "]");
            this.gl.uniform1i(activeLoc, 1);
            let posLoc = this.gl.getUniformLocation(this.prg, "lights[" + i + "].pos");
            this.gl.uniform3fv(posLoc, this.lights[i].pos.arr);
            let strengthLoc = this.gl.getUniformLocation(this.prg, "lights[" + i + "].strength");
            this.gl.uniform1f(strengthLoc, this.lights[i].strength);
            let colorLoc = this.gl.getUniformLocation(this.prg, "lights[" + i + "].color");
            this.gl.uniform3fv(colorLoc, this.lights[i].color.arr);
        }
    }
}

class Camera {
    constructor(pos, fov, gl, prg, canvas) {
        this.pos = pos;
        this.fov = fov;
        this.gl = gl;
        this.prg = prg;
        this.canvas = canvas;
        this.xRotation = 0;
        this.yRotation = 0;
    }
    update() {
        let posLoc = this.gl.getUniformLocation(this.prg, "cameraPos");
        this.gl.uniform3fv(posLoc, this.pos.arr);
        let fovLoc = this.gl.getUniformLocation(this.prg, "fov");
        this.gl.uniform1f(fovLoc, this.fov);
        let widthLoc = this.gl.getUniformLocation(this.prg, "windowWidth") ;
        this.gl.uniform1f(widthLoc, this.canvas.width);
        let heightLoc = this.gl.getUniformLocation(this.prg, "windowHeight") ;
        this.gl.uniform1f(heightLoc, this.canvas.height);
        let xRotationLoc = this.gl.getUniformLocation(this.prg, "cameraXRotation") ;
        this.gl.uniform1f(xRotationLoc, this.xRotation);
        let yRotationLoc = this.gl.getUniformLocation(this.prg, "cameraYRotation") ;
        this.gl.uniform1f(yRotationLoc, this.yRotation);
    }
}

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
    let gl;
    let program;
    function updateGL() {
        let samplesLoc = gl.getUniformLocation(program, "samples");
        gl.uniform1i(samplesLoc, __samples);
        let rayStepsLoc = gl.getUniformLocation(program, "raySteps");
        gl.uniform1i(rayStepsLoc, __raySteps);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        update();
        __objects.update();
        __lights.update();
        camera.update();
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        window.requestAnimationFrame(updateGL);
    }
    load().then(() => {
        canvas = document.querySelector("canvas");
        gl = canvas.getContext("webgl2");

        gl.clearColor(0, 0, 0, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);

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

        let vertexShader = createShader(gl, gl.VERTEX_SHADER, __vertexShaderSource);
        let fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, __fragmentShaderSource);
        program = createProgram(gl, vertexShader, fragmentShader);

        gl.useProgram(program);

        let positionLocation = gl.getAttribLocation(program, "a_position");
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

        /*const numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < numUniforms; ++i) {
            const info = gl.getActiveUniform(program, i);
            console.log(info);
        }*/
        
        __objects = new Objects3D(gl, program);
        __lights = new Lights(gl, program);
        camera = new Camera(new Vector([0, 1, 0]), 90, gl, program, canvas);

        setup();
        updateGL();
    });
}
async function load() {
    __vertexShaderSource = await (await fetch("shader.vert")).text();
    __fragmentShaderSource = await (await fetch("shader.frag")).text();
}