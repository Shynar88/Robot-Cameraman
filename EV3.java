package another;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;
import java.net.UnknownHostException;
import java.rmi.NotBoundException;

import lejos.hardware.motor.Motor;
import lejos.robotics.RegulatedMotor;
import lejos.utility.Delay;
import lejos.hardware.BrickFinder;
import lejos.hardware.Keys;
import lejos.hardware.ev3.EV3;
import lejos.hardware.lcd.TextLCD;

public class Another {
	private static EV3 ev3;
	private static TextLCD lcd;
	private static Keys keys;
	private static String serverAddress;
	private static int serverPort;
	private static Socket socket;
	private static DataOutputStream streamOut;
	private static DataInputStream streamIn;
	private static RegulatedMotor leftMotor;
	private static RegulatedMotor rightMotor;
	
	public static void forward(int dist)
	{
		leftMotor.rotate(dist, true);
		rightMotor.rotate(dist);
	}
	
	public static void backward(int dist)
	{
		leftMotor.rotate(-dist, true);
		rightMotor.rotate(-dist);
	}
	
	public static void left(int angle)
	{
		leftMotor.rotate(-angle, true);
		rightMotor.rotate(angle);
		Delay.msDelay(2000);
	}
	
	public static void right(int angle)
	{
		leftMotor.rotate(angle, true);
		rightMotor.rotate(-angle);
		Delay.msDelay(2000);
	}
	
	public static void main(String[] args) throws IOException, InterruptedException, NotBoundException{
		ev3 = (EV3) BrickFinder.getLocal();
		lcd = ev3.getTextLCD();
		keys = ev3.getKeys();
		
		leftMotor = Motor.D;
		rightMotor = Motor.A;
		
		leftMotor.setSpeed(200);
		rightMotor.setSpeed(200);
		leftMotor.setAcceleration(100);
		rightMotor.setAcceleration(100);
		
		serverAddress = "10.0.1.10";
		serverPort = 8040;
		
		socket = null;
		streamOut = null;
		streamIn = null;
		try{
			lcd.clear();
			lcd.drawString("Waiting...", 1, 1);
			
			socket = new Socket(serverAddress, serverPort);
			lcd.clear();
			lcd.drawString("Connected", 1, 1);
			
			streamIn = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
			streamOut = new DataOutputStream(socket.getOutputStream());
		}catch(UnknownHostException uhe) {
			lcd.clear();
			lcd.drawString("Host unknown: "+uhe.getMessage(), 1, 1);
		}
		
		String sendM = "";
		byte[] recvB = new byte[100];
		String recvM = "";
		while(keys.getButtons() != Keys.ID_ESCAPE) {
			try {
				sendM = "test";
				streamOut.write(sendM.getBytes("UTF-8"));
				streamOut.flush();
				
				streamIn.read(recvB);
				recvM = new String(recvB, "UTF-8");
				lcd.clear();
				lcd.drawString(recvM, 1, 1);
				recvM = recvM.substring(0, 4);
				lcd.drawString(recvM, 1, 3);
				
				if(recvM.equals("forw")) forward(1000);
				else if(recvM.equals("back")) backward(1000);
				else if(recvM.equals("left")) left(50);
				else if(recvM.equals("righ")) right(50);
				
				for(int i = 0; i < 100; i++)
					recvB[i] = 0;
			}catch(IOException ioe) {
				lcd.drawString("Sending error: "+ioe.getMessage(), 1, 4);
			}
		}
		
		if (socket != null) socket.close();
		if (streamOut != null) streamOut.close();
		if (streamIn != null) streamIn.close();
	}
}