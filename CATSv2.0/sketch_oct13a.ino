#include"ServoTimer2.h"

ServoTimer2 serX;
ServoTimer2 serY;
int LED = 12;


String serialData;
int x;

const int trigger=8; 
const int echo=7; 
float dist;
volatile bool programStarted;

void setup() {

  serX.attach(11);
  serY.attach(10);
  pinMode(LED, OUTPUT);
  Serial.begin(9600);
  Serial.setTimeout(20);
  programStarted = false;
  
  cli();// disable the global interrupts system in order to
  //setup the timers
  TCCR1A = 0; // SET TCCR1A and B to 0
  TCCR1B = 0;
  // Set the OCR1A with the desired value:
  OCR1A = 15624;
  // Active CTC mode:
  TCCR1B |= (1 << WGM12);
  // Set the prescaler to 1024:
  TCCR1B |= (1 << CS10);
  TCCR1B |= (1 << CS12);
  // Enable the Output Compare interrupt (by setting the
  //mask bit)
  TIMSK1 |= (1 << OCIE1A);

  sei(); // enable global interrupts
  
  pinMode(trigger,OUTPUT);
  pinMode(echo,INPUT);
}

void loop() {

 digitalWrite(trigger,LOW);
  delayMicroseconds(5);        
  
  digitalWrite(trigger,HIGH);   
  delayMicroseconds(10);        
  digitalWrite(trigger,LOW);  
  
   dist=pulseIn(echo,HIGH);

   dist = dist/58;

   if(programStarted){
    if(dist < 10.0)
      TCNT1 = 0; 
  }
  else{
    digitalWrite(LED, LOW);
    if(dist > 10.0)
      TCNT1 = 0; 
  }

    
}


ISR(TIMER1_COMPA_vect)
{
  if (programStarted){
    programStarted = false;
    Serial.println('0');
    
  }
  else{
    programStarted = true;
    Serial.flush();
    Serial.println('1');
  }
}


void serialEvent() {
  if(programStarted){
    
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
