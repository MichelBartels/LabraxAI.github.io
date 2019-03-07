function intToFract(int) {
    return nerdamer(int + "/" + 1);
}

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementsByTagName("form")[0];
    const input = document.querySelector("form input[type=text]");
    const header = document.querySelector("table thead");
    const contents = document.querySelector("#content");
    const arrows = document.querySelector("#arrow");
    const extrema = document.querySelector("#extremum");
    const calculations = document.querySelector("#calculation");
    form.onsubmit = event => {
        form.style["display"] = "none";
        event.preventDefault();
        const y = nerdamer(input.value.replace(",", ".");
        document.getElementById("y").innerHTML = "$$\\mathrm{f}(x)=" + y.toTeX() + "$$";
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, "y"]);
        const y_ = nerdamer.diff(y, "x");
        document.getElementById("y_").innerHTML = "$$\\mathrm{f}^{\\prime}(x)=" + y_.toTeX() + "$$";
        MathJax.Hub.Queue(["Typeset", MathJax.Hub, "y_"]);
        let solution_i = nerdamer.solve(y_, "x");
        solution_i = solution_i.symbol.elements;
        var solution = [];
        for (let s of solution_i) {
            if (!s.value.includes("i")) {
                solution.push(s);
            }
        }
        a = solution;
        console.log(solution);
        //console.log(nerdamer.);
        
        solution.sort();
        let gradients = [];
        let xs = [];
        for (let i = 0; i <= solution.length * 2; i++) {
            const head = document.createElement("th");
            let x;
            if (i == 0) {
                console.log(solution[i]);
                head.innerHTML = "$$x < " + nerdamer(solution[i]).toTeX() + "$$";
                if (solution[i] > 0) {
                    x = intToFract(0);
                } else {
                    x = intToFract(Math.round(solution[i]) - 1);
                }
            } else if (i % 2 == 1) {
                head.innerHTML = "$$x=" + nerdamer(solution[(i - 1) / 2]).toTeX() + "$$";
                x = solution[(i - 1) / 2];
            } else if (i == solution.length * 2) {
                head.innerHTML = "$$" + nerdamer(solution[i / 2 - 1]).toTeX() + " < x$$";
                if (solution[i / 2 - 1] < 0) {
                    x = intToFract(0);
                } else {
                    x = intToFract(Math.round(solution[i / 2 - 1]) + 1);
                }
            } else {
                head.innerHTML = "$$" + nerdamer(solution[i / 2 - 1]).toTeX() + " < x < " + nerdamer(solution[i / 2]).toTeX() + "$$";
                if (solution[i / 2 - 1] < 0 && solution[i / 2] > 0) {
                    x = intToFract(0);
                } else if ((Math.floor(solution[i / 2 - 1]) + 1) < solution[i] && (solution[i / 2 - 1] % 1) != 0) {
                    x = intToFract(Math.floor(solution[i / 2 - 1]) + 1);
                } else if (Math.floor(solution[i / 2]) > solution[i / 2 - 1] && (solution[i / 2] % 1) != 0) {
                    x = intToFract(Math.floor(solution[i / 2]));
                } else {
                    x = nerdamer(solution[i / 2 - 1]).add(nerdamer(solution[i / 2])).divide(2);
                }
            }
            head.setAttribute("id", "tableHead" + i);
            header.appendChild(head);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, "tableHead" + i]);
            const content = document.createElement("td");
            const y__ = y_.evaluate({"x": x});
            content.innerHTML = "$$\\mathrm{f}^{\\prime}(" + nerdamer(x).toTeX() + ")=" + y__.toTeX() + "$$";
            content.setAttribute("id", "tableContent" + i);
            contents.appendChild(content);
            MathJax.Hub.Queue(["Typeset", MathJax.Hub, "tableContent" + i]);
            const arrow = document.createElement("td");
            let f = parseFloat(y__.toString());
            if (f < 0) {
                arrow.innerHTML = "&#8600;";
                gradients.push(-1);
            } else if (f > 0) {
                arrow.innerHTML = "&#8599;";
                gradients.push(1);
            } else if (f == 0) {
                arrow.innerHTML = "&#8594;";
                gradients.push(0);
            }
            xs.push(x);
            arrows.appendChild(arrow);
        }
        console.log(xs);
        for (let i = 0; i < gradients.length; i++) {
            if (gradients[i] == 0) {
                const extremum = document.createElement("td");
                extremum.setAttribute("id", "extremum" + i);
                const calculation = document.createElement("span");
                console.log(xs[i]);
                let y_ = y.evaluate({"x": xs[i]});
                calculation.innerHTML = "$$\\mathrm{f}(" + nerdamer(xs[i]).toTeX() + ")=" + nerdamer(y_).toTeX() + "$$";
                calculation.setAttribute("id", "calculation" + i);
                if (gradients[i - 1] < 0 && gradients[i + 1] > 0) {
                    extremum.innerHTML = "$$Tiefpunkt(" + nerdamer(xs[i]).toTeX() + "|" + nerdamer(y_).toTeX() + ")$$";
                } else if (gradients[i - 1] > 0 && gradients[i + 1] < 0) {
                    extremum.innerHTML = "$$Hochpunkt(" + nerdamer(xs[i]).toTeX() + "|" + nerdamer(y_).toTeX() + ")$$"
                } else {
                    extremum.innerHTML = "$$Sattelpunkt(" + nerdamer(xs[i]).toTeX() + "|" + nerdamer(y_).toTeX() + ")$$"
                }
                calculations.appendChild(calculation);
                extrema.appendChild(extremum);
                MathJax.Hub.Queue(["Typeset", MathJax.Hub, "calculation" + i]);
                MathJax.Hub.Queue(["Typeset", MathJax.Hub, "extremum" + i]);
            } else {
                extrema.appendChild(document.createElement("td"));
            }
        }
        return false;
    };
});
