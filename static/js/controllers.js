function trainClassifier(){
    if (dataset.length() == 0) {
      alert('Dataset is empty!');
      return;
    }

    if(!dataset.valid()) {
        alert('Create data from boths CLASSes!');
        return;
    } 

    var formData = Object.assign({}, 
                    {'dataset': dataset.getData()}, 
                    getParams()
                  );

    $.ajax({
      type: "POST",
      url: "/api/train-data",
      data: JSON.stringify(formData),
      success: trainSuccessFunction,
      dataType: "json",
      contentType : "application/json"
    });

  }

  function trainSuccessFunction(data){
    clearCanvas(canvasElem);
    reDrawPixel(canvasElem);
    drawDecisionLine(canvasElem, data['points']);
    console.log('Train Complete')
  }

  function reDrawPixel(canvas){
    let X1 = dataset.getData()['x1'];
    let X2 = dataset.getData()['x2'];
    let Y  = dataset.getData()['y'];

    for(i = 0; i < X1.length; i++) {
        drawCoordinates(canvas, X1[i], X2[i], 3.5, Y[i]);
    }
  }
  
  function drawDecisionLine(canvas, data){

    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(data[0][0], data[0][1]);
    ctx.lineTo(data[1][0], data[1][1]);
    ctx.stroke();
    
  }

  function getParams() {
    max_iter = Number(document.getElementById('max_iter').value);
    lrate = lrate_value[document.getElementById('lrate').value];
    C = lrate_value[document.getElementById('C').value];
    return {
      'max_iter'   : max_iter,
      'learning_rate': lrate,
      'C'            : C
    }
  }