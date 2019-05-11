let keysPressed = {
    "w": false,
    "a": false,
    "s": false,
    "d": false,
    "space": false
}
let sphere;
let speed = 0.05;
let vel;

function setup() {
    sphere = new Sphere(new Vector([0, 1, 5]), new Material(new Vector([0, 1, 0]), 1, 0), 2);
    new Sphere(new Vector([1, 1, 2]), new Material(new Vector([1, 0, 0]), 1, 0), 1);
    new Plane(new Vector([0, 1, 0]), new Material(new Vector([1, 1, 1]), 1, 0));
    new Light(new Vector([-3, 1, 0]), 0.2, new Vector([1, 1, 1]));
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
    sphere.pos = sphere.pos.add(new Vector([0, 0.001, 0]));
    if (keysPressed.w) {
        camera.pos = camera.pos.add(new Vector([0, 0, speed]));
    }
    if (keysPressed.s) {
        camera.pos = camera.pos.add(new Vector([0, 0, -speed]));
    }
    if (keysPressed.a) {
        camera.pos = camera.pos.add(new Vector([-speed, 0, 0]));
    }
    if (keysPressed.d) {
        camera.pos = camera.pos.add(new Vector([speed, 0, 0]));
    }
    if (keysPressed.space) {
        vel = 0.5;
    }
    camera.pos = camera.pos.add(new Vector([0, vel, 0]));
    vel -= 0.02;
    if (camera.pos.arr[1] < 1) {
        camera.pos.arr[1] = 1;
        vel = 0;
    }
}