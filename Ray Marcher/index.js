window.onload = () => {
    let canvas = document.querySelector("canvas");
    let ctx = canvas.getContext("2d");
    setup(canvas, ctx);
    function update() {
        let pixels = ctx.createImageData(canvas.width, canvas.height);
        for (let i = 0; i < pixels.data.length; i += 4) {
            let pixelNumber = i / 4;
            let x = pixelNumber % canvas.width;
            let y = (pixelNumber - x) / canvas.width;
            let pixel = renderPixel(x, y);
            pixels.data[i + 0] = pixel[0];
            pixels.data[i + 1] = pixel[1];
            pixels.data[i + 2] = pixel[2];
            pixels.data[i + 3] = 255;
        }
        //renderPixel(0, 0);
        ctx.putImageData(pixels, 0, 0);
        window.requestAnimationFrame(update);
    }
    update();
}

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
    show() {
        console.table(this.arr)
    }
}

class Sphere {
    constructor(pos, color, radius) {
        this.pos = pos;
        this.radius = radius;
        this.color = color;
    }
    getDistance(pos) {
        let sub = this.pos.subtract(pos);
        let len = sub.length;
        return Math.abs(len - this.radius);
    }
    intersects(pos) {}
}

class Light {
    constructor(pos, strength) {
        this.pos = pos;
        this.strength = strength;
    }
    getColor(color, dist) {
        return color.multiply(this.strength / dist);
    }
    getDistance(pos) {
        return Math.abs(pos.subtract(this.pos).length);
    }
}

const maxSteps = 100;
const maxDist  = 100;
const minDist = 0.01;
let objects;
let background = new Vector([0, 0, 0]);
let lights;

function setup(canvas, ctx) {
    objects = [new Sphere(new Vector([0, 0, 5]), new Vector([0, 255, 0]), 2), new Sphere(new Vector([1, 0, 2]), new Vector([255, 0, 0]), 1)];
    lights = [new Light(new Vector([0, 1, 0]), 0.5)];
}

function rayMarch(rayOrigin, rayDirection, objects) {
    let d = 0;
    let intersectObject;
    let pos;
    for (let step = 0; step < maxSteps; step++) {
        pos = rayOrigin.add(rayDirection.multiply(d));
        let newDist;
        for (let object of objects) {
            let dist = object.getDistance(pos);
            if (!newDist || dist < newDist) {
                newDist = dist;
                intersectObject = object;
            }
        }
        d += newDist;
        if (newDist < minDist) {
            return [intersectObject, d, pos];
        }
        if (d > maxDist) {
            return [undefined, maxDist, pos];
        }
    }
    return [undefined, maxDist, pos];
}

function shader(rayOrigin, rayDirection) {
    let object = rayMarch(rayOrigin, rayDirection, objects);
    let intersectPos = object[2];
    object = object[0];
    let color = new Vector([0, 0, 0]);
    let lightsAndObjects = objects.concat(lights);
    for (let i = 0; i < lightsAndObjects.length; i++){ 
        if (lightsAndObjects[i] === object) {
            lightsAndObjects.splice(i, 1); 
        }
     }
    if (object) {
        for (let light of lights) {
            //intersectPos.show();
            //light.pos.show();
            let vec = light.pos.subtract(intersectPos).normalized;
            //vec.show();
            //console.log(vec.length);
            let res = rayMarch(intersectPos, vec, lightsAndObjects);
            if (res[0] == light) {
                color = color.add(light.getColor(object.color, res[1]));
            }
        }
    }
    return color;
    /*if (object[0]) {
        return new Vector([object[1], object[1], object[1]]);
    } else {
        return background;
    }*/
}

function renderPixel(x, y) {
    let rayOrigin = new Vector([0, 1, 0]);
    rayDirection = new Vector([1 - x / 500, 1 - y / 500, 1]).normalized;
    let color = shader(rayOrigin, rayDirection);
    return [color.arr[0], color.arr[1], color.arr[2]];
}