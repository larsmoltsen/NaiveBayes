package moltsen.AI.NaiveBayes.model;

import java.util.List;
import java.util.ArrayList;

public class NaiveBayesData {
	private ArrayList<String> classLabels;
	private ArrayList<Double> priorProbabilities;
	private ArrayList<FeatureData> features;
	
	public ArrayList<FeatureData> getFeatures() {
		return features;
	}
	
	public void setFeatures(ArrayList<FeatureData> features) {
		this.features = features;
	}
	
	public ArrayList<String> getClassLabels() {
		return classLabels;
	}
	
	public void setClassLabels(ArrayList<String> classLabels) {
		this.classLabels = classLabels;
	}

	public ArrayList<Double> getPriorProbabilities() {
		return priorProbabilities;
	}

	public void setPriorProbabilities(ArrayList<Double> priorProbabilities) {
		this.priorProbabilities = priorProbabilities;
	}
}
