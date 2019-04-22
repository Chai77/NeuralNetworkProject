package com.chaitanya.example;

import com.chaitanya.data.TrainingData;
import com.chaitanya.matrix.Matrix;
import com.chaitanya.matrix.MatrixOperations;
import com.chaitanya.neuralnetwork.NeuralNetwork;

public class HandwrittenDigitsNN {
    public static void main(String[] args) {
        int[] layout = new int[]{784, 16, 10};
        NeuralNetwork network = new NeuralNetwork(layout);
        //NeuralNetwork network = new NeuralNetwork("HandwrittenDigits.txt");
        TrainingData training = new TrainingData("train.csv");
        network.train(training, 1000, 256, 0.01);
        network.save("HandwrittenDigits.txt");
        int howMuchData = training.getNumData();
        int numCorrect = 0;
        for(int i = 0; i < 100; i++) {
            NeuralNetwork.IndividualDataSet inputAndOutput = training.get(1, (int)(Math.random() * howMuchData));
            Matrix output = network.test(inputAndOutput.getInputs().get(0));
            MatrixOperations.printMatrix(output, "Output " + i);
            System.out.println();
            Matrix expectedOutput = inputAndOutput.getExpectedOutput();
            int numGot = 0;
            int numExpected = 0;
            for(int k = 0; k < output.getRows(); k++) {
                if(output.getMatrix()[k][0] > output.getMatrix()[numGot][0]) {
                    numGot = k;
                } 
                if(expectedOutput.getMatrix()[k][0] == 1) {
                    numExpected = k;
                }
            }
            if(numGot == numExpected) {
                numCorrect++;
            } else {
                System.out.println("Num got was " + numGot + " while num excpected was " + numExpected);
            }
        }
        System.out.println("The accuracy was " + numCorrect/100.0);
    }
}