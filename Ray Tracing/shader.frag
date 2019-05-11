#version 300 es
precision mediump float;

#define RAY_STEPS 1
#define MIN_DIST 0.01
#define MAX_DIST 1000.
#define MAX_OBJECTS 10
#define MAX_LIGHTS 5
#define MAX_RAY_DEPTH 3
#define PATHS_PER_PIXEL 256
#define EPSILON 1e-3
#define M_PI 3.1415926535897932384626433832795

struct Material {
    vec3 color;
    float diffuse;
    float specular;
};
struct Object {
    vec3 pos;
    float arg1;
    Material material;
    bool sphere;
    bool plane;
};
/*
Sphere: arg1: radius
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
uniform vec3 cameraPos;

vec3 sunDir = normalize(vec3(-0.3,1.3,0.1));
vec3 sunCol = 6.0*vec3(1.0,0.8,0.6);
vec3 skyCol = 4.0*vec3(0.2,0.35,0.5);

Distance getDistance(Object object, Ray ray) {
    if (object.sphere) {
        vec3 rayToSphere = object.pos - ray.rayOrigin;
        float b = dot(ray.rayDirection, rayToSphere);
        float d = b*b - dot(rayToSphere, rayToSphere) + 1.0;
        if (d >= 0.) {
            float dist = b - sqrt(d);
            if (dist >= 0.0) {
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
    return Distance(MAX_DIST, false);
}

vec3 getNormal(Object object, vec3 pos) {
    if (object.sphere) {
        return pos - object.pos;
    }
    if (object.plane) {
        return object.pos;
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
    return Ray(cameraPos, normalize(vec3(x / 500. - 1., y / 500. - 1., 1) * tan(M_PI * fov / 360.)), vec3(0.), Object(vec3(0.), 0., Material(vec3(0.), 0., 0.), false, false), 0., false);
}

out vec4 outColor; 
void main() {
    vec3 color = vec3(0.);
    vec3 mask = vec3(1.);
    Ray ray = generateRay(gl_FragCoord.x, gl_FragCoord.y);
    for (int i = 0; i < RAY_STEPS; i++) {
        ray = traceRay(ray);
        if (ray.stop) {
            break;
        }
        Material material = ray.intersectObject.material;
        vec3 r0 = material.color * material.specular;
        float hv = clamp(dot(ray.normal, -ray.rayDirection), 0., 1.);
        vec3 fresnel = r0 + (1. - r0) * pow(1. - hv, 5.);
        mask *= fresnel;
        vec3 originalOrigin = ray.rayOrigin;
        for (int i = 0; i < lights.length(); i++) {
            vec3 dist = lights[i].pos - ray.rayOrigin;
            float distLength = length(dist);
            vec3 lightDir = normalize(dist);
            ray.rayDirection = lightDir;
            ray.rayOrigin += lightDir * EPSILON;
            if (traceRay(ray).dist > distLength) {
                color += clamp(dot(ray.normal, lightDir), 0., 1.) * lights[i].color * lights[i].strength * material.color * material.diffuse * (1. - fresnel) * mask / fresnel;
            }
        }
        vec3 reflection = reflect(ray.rayDirection, ray.normal);
        ray.rayOrigin = originalOrigin + reflection * EPSILON;
        ray.rayDirection = reflection;
    }
    outColor = vec4(color, 1);
}