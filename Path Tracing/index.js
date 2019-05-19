let keysPressed = {
    "w": false,
    "a": false,
    "s": false,
    "d": false,
    "space": false
}
let speed = 0.05;
let vel;
let jumping = false;
let pointerLocked = false;
let mouseVector;
let time = 0;
let fps;
let raySteps;
let samples;
let currentFpsSinceLastOutput = 0;

function setup() {
    fps = document.getElementById("fps");
    raySteps = document.getElementById("raySteps");
    samples = document.getElementById("sampleSteps");
    mouseVector = new Vector([0, 0]);
    canvas.requestPointerLock = canvas.requestPointerLock || canvas.mozRequestPointerLock;
    canvas.requestPointerLock();
    let clickEvent = () => {
        canvas.requestPointerLock();
    };
    canvas.addEventListener("click", clickEvent);
    let mouseMoved = (event) => {
        mouseVector = mouseVector.add(new Vector([event.movementX, event.movementY]));
    };
    let pointerLockChange = () => {
        if (document.pointerLockElement == canvas) {
            pointerLocked = true;
            document.addEventListener("mousemove", mouseMoved, false);
            canvas.removeEventListener("click", clickEvent);
        } else {
            pointerLocked = false;
            document.removeEventListener("mousemove", mouseMoved, false);
            canvas.addEventListener("click", clickEvent);
        }
    };
    if ("onpointerlockchange" in document) {
        document.addEventListener("pointerlockchange", pointerLockChange, false);
    } else if ("onmozpointerlockchange" in document) {
        document.addEventListener('mozpointerlockchange', pointerLockChange, false);
    }
    //new Plane(new Vector([0, 1, 0]), new Material(new Vector([1, 1, 1]), 1, new Vector([0, 0, 0])));
    let roomSize = 3;
    new Box(new Vector([-roomSize - 1, -1, -roomSize - 1]), new Material(new Vector([0.4, 0.4, 0.4]), 1, new Vector([0, 0, 0])), new Vector([roomSize + 1, 0, roomSize + 1]));
    new Box(new Vector([-roomSize - 1, roomSize, -roomSize - 1]), new Material(new Vector([0.1, 0.1, 0.1]), 1, new Vector([0, 0, 0])), new Vector([roomSize + 1, roomSize + 1, roomSize + 1]));
    new Box(new Vector([-roomSize - 1, 0, -roomSize - 1]), new Material(new Vector([0.160784314, 0.364705882, 0.682352941]), 0.5, new Vector([0, 0, 0])), new Vector([-roomSize, roomSize, roomSize + 1]));
    new Box(new Vector([-roomSize, 0, roomSize]), new Material(new Vector([0.364705882, 0.160784314, 0.682352941]), 0.5, new Vector([0, 0, 0])), new Vector([roomSize, roomSize, roomSize]));
    new Box(new Vector([roomSize, 0, -roomSize - 1]), new Material(new Vector([0.682352941, 0.160784314, 0.364705882]), 0.5, new Vector([0, 0, 0])), new Vector([roomSize + 1, roomSize, roomSize + 1]));
    new Box(new Vector([-roomSize, 0, -roomSize - 1]), new Material(new Vector([0.682352941, 0.364705882, 0.160784314]), 0.5, new Vector([0, 0, 0])), new Vector([roomSize, roomSize, -roomSize]));
    new Light(new Vector([-3, 1, 0]), 0.2, new Vector([1, 1, 1]));
    let lightSize = 0.5;
    new Box(new Vector([-lightSize, roomSize - lightSize, - lightSize]), new Material(new Vector([1, 1, 1]), 1, new Vector([3, 3, 3])), new Vector([lightSize, roomSize + lightSize, lightSize]));
    new Sphere(new Vector([0, 0.5, 0]), new Material(new Vector([1, 0, 0]), 1, new Vector([0, 0, 0])), 0.5);
    new Sphere(new Vector([-1, 1, -1]), new Material(new Vector([0, 1, 0]), 1, new Vector([0, 0, 0])), 1);
    new Sphere(new Vector([-1, 2, 1]), new Material(new Vector([0, 0, 1]), 1, new Vector([0, 0, 0])), 0.75);
    vel = 0;
    document.addEventListener("keydown", (event) => {
        if (event.keyCode == 87) {
            keysPressed.w = true;
        }
        if (event.keyCode == 65) {
            keysPressed.a = true;
        }
        if (event.keyCode == 83) {
            keysPressed.s = true;
        }
        if (event.keyCode == 68) {
            keysPressed.d = true;
        }
        if (event.keyCode == 32) {
            keysPressed.space = true;
        }
    });
    document.addEventListener("keyup", (event) => {
        if (event.keyCode == 87) {
            keysPressed.w = false;
        }
        if (event.keyCode == 65) {
            keysPressed.a = false;
        }
        if (event.keyCode == 83) {
            keysPressed.s = false;
        }
        if (event.keyCode == 68) {
            keysPressed.d = false;
        }
        if (event.keyCode == 32) {
            keysPressed.space = false;
        }
    });
}
function update() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    let movement = new Vector([0, 0, 0]);
    if (keysPressed.w) {
        movement = movement.add(new Vector([0, 0, speed]));
    }
    if (keysPressed.s) {
        movement = movement.add(new Vector([0, 0, -speed]));
    }
    if (keysPressed.a) {
        movement = movement.add(new Vector([-speed, 0, 0]));
    }
    if (keysPressed.d) {
        movement = movement.add(new Vector([speed, 0, 0]));
    }
    if (keysPressed.space && !jumping) {
        vel = 0.5;
        jumping = true;
    }
    camera.pos = camera.pos.add(new Vector([0, vel, 0]));
    vel -= 0.02;
    if (camera.pos.arr[1] < 1) {
        camera.pos.arr[1] = 1;
        vel = 0;
        jumping = false;
    }
    let xAngle = mouseVector.arr[1] / 20;
    let yAngle = mouseVector.arr[0] / 20;
    camera.xRotation = xAngle;
    camera.yRotation = yAngle;
    camera.pos = camera.pos.add(movement.rotateY(yAngle));
    camera.fov = fov.value;
    __raySteps = raySteps.value;
    __samples = samples.value;
    if (currentFpsSinceLastOutput == 60) {
        let now = Date.now();
        fps.innerText = "Fps: " + Math.round(1000 / (now - time) * 60);
        time = now;
        currentFpsSinceLastOutput = 0;
    } else {
        currentFpsSinceLastOutput++;
    }
}