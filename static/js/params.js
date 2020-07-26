var lrate_value = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];
var lrate_input = document.getElementById('lrate'),
    lrate_output = document.getElementById('lrate_out');

lrate_input.oninput = function(){
    lrate_output.innerHTML = lrate_value[this.value];
};
lrate_input.oninput();


var C_value = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 10];
var C_input = document.getElementById('C'),
    C_output = document.getElementById('C_out');

C_input.oninput = function(){
    C_output.innerHTML = C_value[this.value];
};
C_input.oninput();