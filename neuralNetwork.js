var create_sigmoid_function = function(input) {
    if (arguments[1]) {
        return gpu.createKernel(function(number){
            let gradient = number[this.thread.x] * (1 - number[this.thread.x]);
            if (gradient < 0.000001) {
                return 0.000001;
            } else {
                return gradient;
            };
        }).setOutput([input]);
    } else {
        return gpu.createKernel(function(number) {
            return 1 / (1 + Math.exp(-number[this.thread.x]));
        }).setOutput([input]);
    };
};
var create_tanh_function = function(input) {
    if (arguments[1]) {
        return gpu.createKernel(function(number){
            let gradient = 1 - Math.pow(number[this.thread.x], 2);
            if (gradient < 0.0001) {
                return 0.0001
            } else {
                return gradient;
            };
        }).setOutput([input]);
    } else {
        return gpu.createKernel(function(number) {
            return 1 - (2 / (Math.exp(2 * number[this.thread.x]) + 1));
        }).setOutput([input]);
    };
};
var create_relu_function = function(input) {
    if (arguments[1]) {
        return gpu.createKernel(function(number){
            if (number[this.thread.x] > 0) {
                return 1;
            } else {
                return 0;
            };
        }).setOutput([input]);
    } else {
        return gpu.createKernel(function(number) {
            return Math.max(number[this.thread.x], 0);
        }).setOutput([input]);
    };
};
var create_leaky_relu_function = function(input) {
    if (arguments[1]) {
        return gpu.createKernel(function(number){
            if (number[this.thread.x] > 0) {
                return 1;
            } else {
                return 0.01;
            };
        }).setOutput([input]);
    } else {
        return gpu.createKernel(function(number) {
            return Math.max(number[this.thread.x], 0.01 * number[this.thread.x]);
        }).setOutput([input]);
    };
};