class Dataset {
  constructor(){
    this.x1 = [];
    this.x2 = [];
    this.y  = [];
  }
  
  addOne(X1, X2, Y){
    this.x1.push(X1);
    this.x2.push(X2);
    this.y.push(Y);
  }
  
  addBatch(X1, X2, Y){
    this.x1 = X1,
    this.x2 = X2,
    this.y = Y
  }
  
  clearData(){
    this.x1.length = 0;
    this.x2.length = 0;
    this.y.length = 0;
  }
  
  getData() {
    return {
      'x1': this.x1,
      'x2': this.x2,
      'y': this.y
    }
  }

  length() {
    return this.x1.length;
  }

  valid() {
    let temp = new Set(this.y);
    return (temp.size == 2) ? true : false;
  }
}
