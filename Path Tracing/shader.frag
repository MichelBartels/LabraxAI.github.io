#version 300 es
precision mediump float;

#define MAX_DIST 1000.
#define MAX_OBJECTS 10
#define MAX_LIGHTS 5
#define EPSILON 1e-3
#define M_PI 3.1415926535897932384626433832795

struct Material {
    vec3 color;
    float albedo;
    vec3 light;
};
struct Object {
    vec3 pos;
    float arg1;
    vec3 arg2;
    Material material;
    bool sphere;
    bool plane;
    bool box;
};
/*
Sphere: arg1: radius
Box: arg2: pos2
*/
struct Light {
    vec3 pos;
    vec3 color;
    float strength;
};
struct Distance {
    float dist;
    bool intersects;
};
struct Ray {
    vec3 rayOrigin;
    vec3 rayDirection;
    vec3 normal;
    Object intersectObject;
    float dist;
    bool stop;
};

uniform Object objects[MAX_OBJECTS];
uniform bool activeObjects[MAX_OBJECTS];
uniform Light lights[MAX_LIGHTS];
uniform bool activeLights[MAX_LIGHTS];
uniform float fov;
uniform float windowWidth;
uniform float windowHeight;
uniform vec3 cameraPos;
uniform float cameraXRotation;
uniform float cameraYRotation;
uniform int raySteps;
uniform int samples;
uniform vec2 firstSeed;

vec2 seed;

vec2 rand() {
    seed += vec2(-1,1);
    return vec2(fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453),
		fract(cos(dot(seed.xy ,vec2(4.898,7.23))) * 23421.631));
}

vec3 ortho(vec3 v) {
    return abs(v.x) > abs(v.z) ? vec3(-v.y, v.x, 0.0)  : vec3(0.0, -v.z, v.y);
}

vec3 getSample(vec3 dir) {
	dir = normalize(dir);
	vec3 o1 = normalize(ortho(dir));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r = rand();
	r.x = r.x * 2. * M_PI;
	r.y = sqrt(r.y);
	float oneminus = sqrt(1.0 - r.y * r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

Distance getDistance(Object object, Ray ray) {
    if (object.sphere) {
        vec3 rayToSphere = ray.rayOrigin - object.pos;
        float a = dot(ray.rayDirection, ray.rayDirection);
        float b = 2. * dot(rayToSphere, ray.rayDirection);
        float c = dot(rayToSphere, rayToSphere) - object.arg1 * object.arg1;
        float d = b * b - 4. * a * c;
        if (d >= 0.) {
            float dist = (-b - sqrt(d)) / (2. * a);
            if (dist >= 0.) {
                return Distance(dist, true);
            }
        }
    }
    if (object.plane) {
        float len = -dot(ray.rayOrigin, object.pos) / dot(ray.rayDirection, object.pos);
        if (len >= 0.) {
            return Distance(len, true);
        }
    }
    if (object.box) {
        float tnear = -MAX_DIST;
        float tfar = MAX_DIST;
        float t1;
        float t2;
        float tmp;
        for (int i = 0; i < 3; i++) {
            if (ray.rayDirection[i] == 0. && (ray.rayOrigin[i] < object.pos[i] || ray.rayOrigin[i] > object.arg2[i])) {
                return Distance(MAX_DIST, false);
            }
            t1 = (object.pos[i] - ray.rayOrigin[i]) / ray.rayDirection[i];
            t2 = (object.arg2[i] - ray.rayOrigin[i]) / ray.rayDirection[i];
            if (t1 > t2) {
                tmp = t1;
                t1 = t2;
                t2 = tmp;
            }
            if (t1 > tnear) {
                tnear = t1;
            }
            if (t2 < tfar) {
                tfar = t2;
            }
            if (tnear > tfar) {
                return Distance(MAX_DIST, false);
            }
            if (tfar < 0.) {
                return Distance(MAX_DIST, false);
            }
        }
        return Distance(tnear, true);
    }
    return Distance(MAX_DIST, false);
}

vec3 getNormal(Object object, vec3 pos) {
    if (object.sphere) {
        return pos - object.pos;
    }
    if (object.plane) {
        return object.pos;
    }
    if (object.box) {
        /*vec3 c = (object.pos + object.arg2) * 0.5;
        vec3 p = pos - c;
        vec3 d = (object.pos + object.arg2) * 0.5;
        return normalize(vec3(
            float(int(p.x / abs(d.x) * 1.000001)),
            float(int(p.y / abs(d.y) * 1.000001)),
            float(int(p.z / abs(d.z) * 1.000001))
        ));*/
        if (abs(pos.x - object.pos.x) < 0.01) {
            return vec3(-1, 0, 0);
        }
        if (abs(pos.x - object.arg2.x) < 0.01) {
            return vec3(1, 0, 0);
        }
        if (abs(pos.y - object.pos.y) < 0.01) {
            return vec3(0, -1, 0);
        }
        if (abs(pos.y - object.arg2.y) < 0.01) {
            return vec3(0, 1, 0);
        }
        if (abs(pos.z - object.pos.z) < 0.01) {
            return vec3(0, 0, -1);
        }
        if (abs(pos.z - object.arg2.z) < 0.01) {
            return vec3(0, 0, 1);
        }
    }
    return vec3(0);
}

Ray traceRay(Ray ray) {
    float d = MAX_DIST;
    bool foundIntersection = false;
    Object intersectObject;
    for (int i = 0; i < objects.length(); i++) {
        if (!activeObjects[i]) {
            break;
        }
        Distance dist = getDistance(objects[i], ray);
        if (dist.intersects) {
            if (dist.dist < d) {
                d = dist.dist;
                intersectObject = objects[i];
                foundIntersection = true;
            }
        }
    }
    if (foundIntersection) {
        vec3 pos = ray.rayOrigin + ray.rayDirection * d;
        vec3 normal = getNormal(intersectObject, pos);
        return Ray(pos, ray.rayDirection, normal, intersectObject, d, false);
    }
    ray.stop = true;
    ray.dist = MAX_DIST;
    return ray;
}

Ray generateRay(float x, float y) {
    float aspectRatio = windowWidth / windowHeight;
    float angle = tan(radians(fov) / 2.);
    float rotationXAngle = radians(cameraXRotation);
    float sinX = sin(rotationXAngle);
    float cosX = cos(rotationXAngle);
    float rotationYAngle = radians(cameraYRotation);
    float sinY = sin(rotationYAngle);
    float cosY = cos(rotationYAngle);
    x = (2. * x / windowWidth - 1.) * aspectRatio * angle;
    y = (2. * y / windowHeight - 1.) * angle;
    float z = y * sinX + cosX;
    y = y * cosX - sinX;
    return Ray(cameraPos, normalize(vec3(x * cosY + z * sinY, y, -x * sinY + z * cosY)), vec3(0.), Object(vec3(0.), 0., vec3(0.), Material(vec3(0.), 0., vec3(0.)), false, false, false), 0., false);
}

out vec4 outColor; 
void main() {
    seed = firstSeed * gl_FragCoord.xy/vec2(windowWidth, windowHeight);
    vec3 colorSum = vec3(0.);
    Ray startRay = generateRay(gl_FragCoord.x, gl_FragCoord.y);
    for (int i = 0; i < samples; i++) {
        Ray ray = startRay;
        //ray.rayDirection = getRandomDir(ray.rayDirection);
        vec3 color = vec3(1.);
        vec3 light = vec3(0.);
        for (int i = 0; i < raySteps; i++) {
            ray = traceRay(ray);
            if (ray.stop) {
                //light += vec3(1.);
                color = vec3(0.);
                break;
            }
            Material material = ray.intersectObject.material;
            light += color * material.light;
            ray.rayDirection = getSample(ray.normal);
            color *= material.color * material.albedo;
            ray.rayOrigin += ray.rayDirection * EPSILON;
        }
        colorSum += light + color;
    }
    outColor = vec4(colorSum / float(samples), 1);
    //outColor = vec4(rand(), rand().x, 1);
}