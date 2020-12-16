#include<Servo.h>

Servo serX;
Servo serY;
int LED = 12;


String serialData;
boolean activate = false;
int x;
void setup() {

  serX.attach(11);
  serY.attach(10);
  pinMode(LED, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(20);
}

void loop() {
  
    
   
}

void serialEvent() {
  serialData = Serial.readString();
 
  int x = parseDataX(serialData);
  if (x != 0){
    digitalWrite(LED, HIGH);
  }else{
    digitalWrite(LED, LOW);
  }
  //Serial.print(x);
  serX.write(x);
  serY.write(parseDataY(serialData));
   
}

int parseDataX(String data) {
  data.remove(data.indexOf("Y"));
  data.remove(data.indexOf("X"), 1);

  return data.toInt();
  

}

int parseDataY(String data) {
  data.remove(0, data.indexOf("Y") + 1);
  return data.toInt();
}
