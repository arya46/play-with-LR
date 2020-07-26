function clearAll(canvas){	
    dataset.clearData();
    clearCanvas(canvas);
};

function clearCanvas(canvas){
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}
  
function dataGen(){
    $.ajax({
        type: "GET",
        url: "/api/generate-data",
        success: genDataSuccess,
        dataType: "json",
        contentType : "application/json"
    });

    
}

function genDataSuccess(data) {
    clearAll(canvasElem);
    dataset.addBatch(data['x1'], data['x2'], data['y']);
    reDrawPixel(canvasElem);
}

function getCoordinates(canvas){

    let rect = canvas.getBoundingClientRect(); 
    let x = event.clientX - rect.left; 
    let y = event.clientY - rect.top; 

    return [x, y];
}
function drawCoordinates(canvas, x, y, pointSize, c_class, verbose=false){	

    let ctx = canvas.getContext("2d");
    if(verbose){
        console.log(x, y);
    }
        
    ctx.beginPath();
    ctx.arc(x, y, pointSize, 0, Math.PI * 2);

    ctx.fillStyle = color_dict[c_class]; 
    ctx.fill();

}