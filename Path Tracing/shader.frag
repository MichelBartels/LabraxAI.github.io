#version 300 es
precision mediump float;

#define MAX_DIST 1000.
#define MAX_OBJECTS 10
#define MAX_LIGHTS 5
#define EPSILON 1e-3

struct Material {
    vec3 color;
    float albedo;
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
uniform float windowWidth;
uniform float windowHeight;
uniform vec3 cameraPos;
uniform float cameraXRotation;
uniform float cameraYRotation;
uniform int raySteps;
uniform int samples;

vec2 seed;

vec2 rand2n() {
    seed+=vec2(-1,1);
    return vec2(fract(sin(dot(seed.xy ,vec2(12.9898,78.233))) * 43758.5453),
		fract(cos(dot(seed.xy ,vec2(4.898,7.23))) * 23421.631));
}

vec3 ortho(vec3 v) {
    return abs(v.x) > abs(v.z) ? vec3(-v.y, v.x, 0.0)  : vec3(0.0, -v.z, v.y);
}

vec3 getSampleBiased(vec3  dir, float power) {
	dir = normalize(dir);
	vec3 o1 = normalize(ortho(dir));
	vec3 o2 = normalize(cross(dir, o1));
	vec2 r = rand2n();
	r.x=r.x*2.*3.1415926535897932384626433832795;
	r.y=pow(r.y,1.0/(power+1.0));
	float oneminus = sqrt(1.0-r.y*r.y);
	return cos(r.x)*oneminus*o1+sin(r.x)*oneminus*o2+r.y*dir;
}

vec3 getSample(vec3 dir) {
	return getSampleBiased(dir,0.0);
}

vec3 getCosineWeightedSample(vec3 dir) {
	return getSampleBiased(dir,1.0);
}

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
    return Ray(cameraPos, normalize(vec3(x * cosY + sinY, y * cosX - sinX, -x * sinY + (y * sinX + cosX) * cosY)), vec3(0.), Object(vec3(0.), 0., Material(vec3(0.), 0.), false, false), 0., false);
}

out vec4 outColor; 
void main() {
    vec3 color = vec3(0.);
    Ray startRay = generateRay(gl_FragCoord.x, gl_FragCoord.y);
    for (int i = 0; i < samples; i++) {
        Ray ray = startRay;
        vec3 mask = vec3(1.);
        for (int i = 0; i < raySteps; i++) {
            ray = traceRay(ray);
            if (ray.stop) {
                color += mask;
                break;
            }
            Material material = ray.intersectObject.material;
            mask *= material.color * 0.5;
            ray.rayDirection = getCosineWeightedSample(ray.normal);
            ray.rayOrigin += ray.rayDirection * EPSILON;
        }
    }
    outColor = vec4(color / float(samples), 1);
}