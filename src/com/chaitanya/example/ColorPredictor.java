package com.chaitanya.example;

import java.util.ArrayList;
import java.util.Arrays;

import com.chaitanya.data.TrainingData;
import com.chaitanya.matrix.Matrix;
import com.chaitanya.neuralnetwork.NeuralNetwork;

public class ColorPredictor {

    public static void main(String[] args) {
        int[] layout = new int[]{3, 5, 1};
        NeuralNetwork network = new NeuralNetwork(layout);
        //r + g + b >= 300
        double[][][] inputs = new double[256 * 256 * 256][3][1];
        double[][][] outputs = new double[256*256*256][1][1];
        int count = 0;
        for(int r = 0; r < 256; r++) {
            for(int g = 0; g < 256; g++) {
                for(int b = 0; b < 256; b++) {
                    inputs[count][0][0] = r/255.0;
                    inputs[count][1][0] = g/255.0;
                    inputs[count][2][0] = b/255.0;
                    outputs[count][0][0] = (r + g + b > 300) ? 0 : 1;
                }
            }
        }
        Matrix[] inputsMat = new Matrix[256 * 256 * 256];
        Matrix[] outputMat = new Matrix[256 * 256 * 256];
        for(int i = 0; i < inputs.length; i++) {
            inputsMat[i] = new Matrix(inputs[i]);
            outputMat[i] = new Matrix(outputs[i]);
        }
        ArrayList<Matrix> inputAL = new ArrayList<>(Arrays.asList(inputsMat));
        ArrayList<Matrix> expectedOutputsAL = new ArrayList<>(Arrays.asList(outputMat));
        TrainingData data = new TrainingData(inputAL, expectedOutputsAL);
        network.train(data, 1000, 256, 0.1);
        network.save("ColorPredictor.txt");
    }
}