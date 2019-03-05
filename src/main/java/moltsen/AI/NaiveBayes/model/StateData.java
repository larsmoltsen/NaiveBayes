package moltsen.AI.NaiveBayes.model;

import java.util.ArrayList;

/**
 * The data representation of a feature state.
 * 
 * @author  Lars Moltsen
 * @version 1.0
 */
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
