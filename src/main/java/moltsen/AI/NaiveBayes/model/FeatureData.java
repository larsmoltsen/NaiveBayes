package moltsen.AI.NaiveBayes.model;

import java.util.ArrayList;

public class FeatureData {
	private String name;
	private ArrayList<StateData> states;
	
	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public ArrayList<StateData> getStates() {
		return states;
	}
	
	public void setStates(ArrayList<StateData> states) {
		this.states = states;
	}
}
