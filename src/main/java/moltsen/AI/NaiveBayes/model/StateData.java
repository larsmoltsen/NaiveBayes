package moltsen.AI.NaiveBayes.model;

import java.util.ArrayList;

public class StateData {
	private String label;
	private ArrayList<Double> conditionalProbabilities;

	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	public ArrayList<Double> getConditionalProbabilities() {
		return conditionalProbabilities;
	}

	public void setConditionalProbabilities(ArrayList<Double> conditionalProbabilities) {
		this.conditionalProbabilities = conditionalProbabilities;
	}
}
