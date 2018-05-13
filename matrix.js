const GPU_ENABLED = true;
function new_random_array(number_of_items, min, max) {
    let array = new Float32Array(number_of_items);
    let factor = (max - min);
    for (let i = 0; i < number_of_items; i++) {
        array[i] = Math.random() * factor + min;
    };
    return array;
};
function new_normally_distributed_array(number_of_items, mean, standard_deviation) {
    let array = new Float32Array(number_of_items);
    let tau = 2 * Math.PI;
    for (let i = 0; i < number_of_items; i++) {
        array[i] = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(tau * Math.random()) * standard_deviation + mean;
    };
    return array;
};
if (GPU_ENABLED) {
    var gpu = new GPU();
    var create_multiply_function = function(m, mn, n) {
        if (!isNaN(n)) {
            let kernel = gpu.createKernel(function(matrix1, matrix2) {
                let value = 0;
                for (let i = 0; i < this.constants.mn; i++) {
                    value += matrix1[mn * Math.floor(this.thread.x / this.constants.n) + i] * matrix2[this.constants.n * i + (this.thread.x % this.constants.n)];
                };
                return value;
            }, {
                constants: {
                    m: m,
                    mn: mn,
                    n: n
                }
            }).setOutput([m * n]);
            /* return function(matrix1, matrix2) {
                return new Matrix(kernel(matrix1.array, matrix2.array), m, n);
            }; */
            return kernel;
        } else if (n == "scalar") {
            n = mn;
            let kernel = gpu.createKernel(function(matrix) {
                return matrix[this.thread.x] * this.constants.number;
            }, {
                constants:{
                    number: arguments[3]
                }
            }).setOutput([m * n]);
            /* return function(matrix, number) {
                return new Matrix(kernel(matrix.array, number), m, n);
            }; */
            return kernel;
        } else if (n == "matrix") {
            n = mn;
            let kernel = gpu.createKernel(function(matrix1, matrix2) {
                return matrix1[this.thread.x] * matrix2[this.thread.x];
            }).setOutput([m * n]);
            /* return function(matrix, number) {
                return new Matrix(kernel(matrix.array, number), m, n);
            }; */
            return kernel;
        };
    };
    var create_matrix_vector_add_function = function(m, n) {
        let kernel = gpu.createKernel(function(matrix, vector) {
            return matrix[this.thread.x] + vector[this.thread.x % this.constants.n];
        }, {
            constants: {
                n: n
            }
        }).setOutput([m * n]);
        /* return function(matrix, vector) {
            return new Matrix(kernel(matrix.array, vector.array), m, n);
        }; */
        return kernel;
    };
    var create_matrix_vector_subtract_function = function(m, n) {
        let kernel = gpu.createKernel(function(matrix, vector) {
            return matrix[this.thread.x] - vector[this.thread.x % this.constants.n];
        }, {
            constants: {
                n: n
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_matrix_matrix_add_function = function(m, n) {
        let kernel = gpu.createKernel(function(matrix1, matrix2) {
            return matrix1[this.thread.x] + matrix2[this.thread.x];
        }).setOutput([m * n]);
        /* return function(matrix, vector) {
            return new Matrix(kernel(matrix.array, vector.array), m, n);
        }; */
        return kernel;
    };
    var create_matrix_matrix_subtract_function = function(m, n) {
        let kernel = gpu.createKernel(function(matrix1, matrix2) {
            return matrix1[this.thread.x] - matrix2[this.thread.x];
        }).setOutput([m * n]);
        /* return function(matrix, vector) {
            return new Matrix(kernel(matrix.array, vector.array), m, n);
        }; */
        return kernel;
    };
/*     var create_elementwise_function = function(m, n, function_) {
        console.log(function_);
        gpu.addFunction(function_, Number, Number);
        let kernel = gpu.createKernel(function(matrix) {
            return function_(matrix[this.thread.x]);
        })
        .setOutput([m * n]);
        return function(matrix) {
            return new Matrix(kernel(matrix.array), m, n);
        };
    }; */
    var create_sum_function = function(m, n) {
        let kernel = gpu.createKernel(function(matrix) {
            let sum = 0;
            for (let i = 0; i < this.constants.m; i++) {
                sum += matrix[this.constants.n * i + this.thread.x]
            };
            return sum;
        }, {
            constants: {
                m: m,
                n: n
            }
        }).setOutput([n]);
        return kernel;
    };
    var create_log_function = function(m, n) {
        let kernel = gpu.createKernel(function(matrix) {
            return Math.log(matrix[this.thread.x]);
        }).setOutput([m * n]);
        return kernel
    };
    var create_division_function = function(m, n, dividend) {
        let kernel = gpu.createKernel(function(matrix) {
            return this.constants.dividend / matrix[this.thread.x];
        }, {
            constants: {
                dividend: dividend
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_subtract_from_scalar_function = function(m, n, scalar) {
        let kernel = gpu.createKernel(function(matrix) {
            return this.constants.scalar - matrix[this.thread.x];
        }, {
            constants: {
                scalar: scalar
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_subtract_scalar_function = function(m, n, scalar) {
        let kernel = gpu.createKernel(function(matrix) {
            return matrix[this.thread.x] - this.constants.scalar;
        }, {
            constants: {
                scalar: scalar
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_add_scalar_function = function(m, n, scalar) {
        let kernel = gpu.createKernel(function(matrix) {
            return matrix[this.thread.x] + this.constants.scalar;
        }, {
            constants: {
                scalar: scalar
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_first_moment_update_function = function(m, n, beta_1) {
        let kernel = gpu.createKernel(function(m_old, gradient) {
            return this.constants.beta_1 * m_old[this.thread.x] + (1 - beta_1) * gradient[this.thread.x];
        }, {
            constants: {
                beta_1: beta_1
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_second_moment_update_function = function(m, n, beta_2) {
        let kernel = gpu.createKernel(function(v_old, gradient) {
            return this.constants.beta_2 * v_old[this.thread.x] + (1 - beta_2) * Math.pow(gradient[this.thread.x], 2);
        }, {
            constants: {
                beta_2: beta_2
            }
        }).setOutput([m * n]);
        return kernel;
    };
    var create_bias_correction_function = function(m, n) {
        let kernel = gpu.createKernel(function(m_old, beta) {
            return m_old[this.thread.x] / (1 - beta)
        }).setOutput([m * n]);
        return kernel;
    };
    var create_theta_update_function = function(m, n, alpha, epsilon) {
        let kernel = gpu.createKernel(function(old_theta, m, v) {
            return /*old_theta[this.thread.x] - */this.constants.alpha * m[this.thread.x] / (Math.sqrt(v[this.thread.x]) + this.constants.epsilon);
        }, {
            constants: {
                alpha: alpha,
                epsilon: epsilon
            }
        }).setOutput([m * n]);
        return kernel;
    };
};
class Matrix {
    constructor() {
        if (arguments.length == 1) {
            this.m = arguments[0].length;
            this.n = arguments[0][0].length;
            this.array = new Float32Array([].concat.apply([], arguments[0]));
        } else if (arguments.length == 2 || arguments.length == 3 & typeof arguments[0] == "number") {
            this.m = arguments[0];
            this.n = arguments[1];
            this.array = new Float32Array(this.m * this.n);
            if (arguments.length == 3) {
                this.array.fill(arguments[2]);
            };
        } else if (arguments.length == 3 & typeof arguments[0] != "number") {
            this.m = arguments[1];
            this.n = arguments[2]
            this.array = arguments[0]
        };
        return new Proxy(this, {
            get(object, property) {                                
                if (property.includes(",")) {
                    property = JSON.parse("[" + property + "]");
                    return object.get_item(property[0], property[1]);
                } else {
                    return object[property];
                }
            },
            set(object, property, value) {
                if (property.includes(",")) {
                    property = JSON.parse("[" + property + "]");
                    return object.set_item(property[0], property[1], value);
                } else {
                    return object[property] = value;
                };
            }
        });
    };
    get multidimensional_array() {
        let new_array = [];
        for (let i = 0; i < this.m; i++) {
            let beginning = i * this.n;
            new_array.push(this.array.subarray(beginning, beginning + this.n));
        };
        return new_array;
    };
    get transpose_matrix() {
        let new_array = new Float32Array(this.m * this.n);
        for (let m = 0; m < this.m; m++) {
            for (let n = 0; n < this.n; n++) {
                new_array[this.m * n + m] = this.array[this.n * m + n];
            };
        };
        return new Matrix(new_array, this.n, this.m);
    };
    get_item(m, n) {
        return this.array[this.n * m + n];
    };
    set_item(m, n, value) {
        return this.array[this.n * m + n] = value;
    };
    show() {
        console.table(this.multidimensional_array);
    };
    transpose() {
        let new_array = new Float32Array(this.m * this.n);
        for (let m = 0; m < this.m; m++) {
            for (let n = 0; n < this.n; n++) {
                new_array[this.m * n + m] = this.array[this.n * m + n];
            };
        };
        this.array = new_array;
        let m = this.m;
        this.m = this.n;
        this.n = m;
    };
    multiply(matrix) {
        if (matrix instanceof Matrix) {
            let new_array = new Float32Array(this.m * matrix.n);
            for (let m = 0; m < this.m; m++) {
                for (let n = 0; n < matrix.n; n++) {
                    for (let mn = 0; mn < this.n; mn++) {
                        new_array[matrix.n * m + n] += this.array[this.n * m + mn] * matrix.array[matrix.n * mn + n];
                    };
                };
            };
            return new Matrix(new_array, this.m, matrix.n);
        } else {
            let factor = matrix;
            let new_array = new Float32Array(this.m * this.n);
            for (let m = 0; m < this.m; m++) {
                for (let n = 0; n < this.n; n++) {
                    new_array[this.n * m + n] = this.array[this.n * m + n] * number;
                }
            }
        }
    };
    add(matrix) {
        if (matrix instanceof Matrix) {
            let new_array = new Float32Array(this.m * this.n);
            for (let m = 0; m < this.m; m++) {
                for (let n = 0; n < this.n; n++) {
                    new_array[this.n * m + n] = this.array[this.n * m + n] + matrix.array[this.n * m + n];
                };
            };
            return new Matrix(new_array, this.m, this.n);
        } else {
            let vector = matrix;
            let new_array = new Float32Array(this.m * this.n);
            for (let m = 0; m < this.m; m++) {
                for (let n = 0; n < this.n; n++) {
                    new_array[this.n * m + n] = this.array[this.n * m + n] + vector.array[n];
                };
            };
            return new Matrix(new_array, this.m, this.n);
        };
    };
    subtract(matrix) {
        if (matrix instanceof Matrix) {
            let new_array = new Float32Array(this.m * this.n);
            for (let m = 0; m < this.m; m++) {
                for (let n = 0; n < this.n; n++) {
                    new_array[this.n * m + n] = this.array[this.n * m + n] - matrix.array[this.n * m + n];
                };
            };
            return new Matrix(new_array, this.m, this.n);
        } else {
            let vector = matrix;
            let new_array = new Float32Array(this.m * this.n);
            for (let m = 0; m < this.m; m++) {
                for (let n = 0; n < this.n; n++) {
                    new_array[this.n * m + n] = this.array[this.n * m + n] - vector.array[n];
                };
            };
            return new Matrix(new_array, this.m, this.n);
        };
    };
    apply_function(function_) {
        let total = m * n;
        let new_array = new Float32Array(total);
        for (let i = 0; i < total; total++) {
            new_array[i] = function_(this.array[i]);
        };
        return new Matrix(new_array);
    };
    get shape() {
        return {
            m: this.m,
            n: this.n
        };
    };
    get mean_squared() {
        let total = 0;
        for (let i = 0; i < this.array.length; i++) {
            total += Math.pow(this.array[i], 2);
        };
        return total / this.array.length;
    };
    get mean() {
        let total = 0;
        for (let i = 0; i < this.array.length; i++) {
            total += this.array[i];
        };
        return total / this.array.length;
    };
    sum() {
        let new_array = new Float32Array(this.n);
        for (let m = 0; m < this.m; m++) {
            for (let n = 0; n < this.n; n++) {
                new_array[n] += this.array[this.n * m + n];
            };
        };
        return new Vector(new_array);
    };
};
class Vector {
    constructor() {
        if (arguments.length == 1) {
            if (typeof arguments[0] == "number") {
                this.dimensions = arguments[0];
                this.array = new Float32Array(this.dimensions);
            } else {
                this.dimension = arguments[0].length;
                this.array = arguments[0];
            }
        } else if (arguments.length == 2) {
            this.dimensions = arguments[0];
            this.array = new Float32Array(this.dimensions);
            this.array.fill(arguments[1]);
        };
        return new Proxy(this, {
            get(object, property) {                                
                if (!isNaN(property)) {
                    property = parseInt(property);
                    return object.get_item(property);
                } else {
                    return object[property];
                }
            },
            set(object, property, value) {
                if (!isNaN(property)) {
                    property = parseInt(property);
                    return object.set_item(property, value);
                } else {
                    return object[property] = value;
                };
            }
        });
    };
    show() {
        console.table(this.array);
    };
    get_item(index) {
        return this.array[index];
    };
    set_item(index, value) {
        return this.array[index] = value;
    };
    get shape() {
        return {m: this.m};
    }
};