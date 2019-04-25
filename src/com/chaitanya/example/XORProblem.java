package com.chaitanya.example;

import com.chaitanya.data.TrainingData;
import java.util.ArrayList;
import java.util.Arrays;
import com.chaitanya.matrix.Matrix;
import com.chaitanya.neuralnetwork.NeuralNetwork;;

public class XORProblem {
    public static void main(String[] args) {
        int[] layout = new int[]{2, 3, 4, 1};
        NeuralNetwork network = new NeuralNetwork(layout);
        //NeuralNetwork network = new NeuralNetwork("XORProblem.txt");
        //Matrix layer1W = new Matrix(new double[][]{{0.1, 0.3},{0.2, 0.5},{0.6, 0.9}});
        //Matrix layer2W = new Matrix(new double[][]{{0.3, 0.7, 0.8}});
        //Matrix layer1B = new Matrix(new double[][]{{0.3}, {0.5}, {0.1}});
        //Matrix layer2B = new Matrix(new double[][]{{0.4}});
        //Matrix[] biases = new Matrix[]{layer1B, layer2B};
        //Matrix[] weights = new Matrix[]{layer1W, layer2W};
        //network.setWeights(weights);
        //network.setBiases(biases);
        network.printWeights();
        network.printBiases();
        Matrix testCase1 = new Matrix(new double[][]{{1}, {1}});
        Matrix testCase2 = new Matrix(new double[][]{{1}, {0}});
        Matrix testCase3 = new Matrix(new double[][]{{0}, {1}});
        Matrix testCase4 = new Matrix(new double[][]{{0}, {0}});
        Matrix testCase1O = new Matrix(new double[][]{{0}});
        Matrix testCase2O = new Matrix(new double[][]{{1}});
        Matrix testCase3O = new Matrix(new double[][]{{1}});
        Matrix testCase4O = new Matrix(new double[][]{{0}});
        Matrix[] input = new Matrix[]{testCase1, testCase2, testCase3, testCase4};
        Matrix[] expectedOutputs = new Matrix[]{testCase1O, testCase2O, testCase3O, testCase4O};
        ArrayList<Matrix> inputAL = new ArrayList<>(Arrays.asList(input));
        ArrayList<Matrix> expectedOutputsAL = new ArrayList<>(Arrays.asList(expectedOutputs));
        TrainingData data = new TrainingData(inputAL, expectedOutputsAL);
        System.out.println(network.test(testCase1));
        System.out.println(network.test(testCase2));
        System.out.println(network.test(testCase3));
        System.out.println(network.test(testCase4));
        network.train(data, 1000000, 1, 0.1);
        //network.save("XORProblem.txt");
        System.out.println(network.test(testCase1));
        System.out.println(network.test(testCase2));
        System.out.println(network.test(testCase3));
        System.out.println(network.test(testCase4));
    }
}