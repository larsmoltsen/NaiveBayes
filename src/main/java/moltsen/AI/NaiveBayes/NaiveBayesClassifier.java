package moltsen.AI.NaiveBayes;

import java.util.ArrayList;
import java.util.Map;

import moltsen.AI.NaiveBayes.model.*;

public class NaiveBayesClassifier {
	private NaiveBayesData data;

	
	/**
	 * Add a class label.
	 * 
	 * @param label
	 * @throws DataStructureException 
	 */
	public void addLabel(String label) throws DataStructureException {
		if (data == null) { initNaiveBayesClassifier(); }
		if (data.getClassLabels().contains(label)) {	throw new DataStructureException("Label already exists (\"" + label + "\")"); }
		
		data.getClassLabels().add(label);
		data.getPriorProbabilities().add(1d);
		for (FeatureData feature : data.getFeatures()) {
			for (StateData state : feature.getStates()) {
				state.getConditionalProbabilities().add(1d);
			}
		}
	}
	
	
	/**
	 * Removes the specified label.
	 * 
	 * @param label
	 * @throws DataStructureException
	 */
	public void removeLabel(String label) throws DataStructureException {
		if (data == null) { throw new DataStructureException("Label does not exist (\"" + label + "\")"); }

		int i = data.getClassLabels().indexOf(label);
		if (i == -1) { throw new DataStructureException("Label does not exist (\"" + label + "\")"); }

		data.getClassLabels().remove(i);
		data.getPriorProbabilities().remove(i);
		for (FeatureData feature : data.getFeatures()) {
			for (StateData state : feature.getStates()) {
				state.getConditionalProbabilities().remove(i);
			}
		}				
	}
	
	
	/**
	 * Adds a feature.
	 * 
	 * @param name
	 * @throws DataStructureException
	 */
	public void addFeature(String featureName) throws DataStructureException {
		if (data == null) { initNaiveBayesClassifier(); }
		if (indexOfFeature(featureName) >= 0) { throw new DataStructureException("Feature already exists (\"" + featureName + "\")"); }
		
		FeatureData newFD = new FeatureData();
		newFD.setName(featureName);
		newFD.setStates(new ArrayList<StateData>());
		data.getFeatures().add(newFD);
	}
	
	
	/**
	 * Remove a feature.
	 * 
	 * @param name
	 * @throws DataStructureException
	 */
	public void removeFeature(String featureName) throws DataStructureException {
		int i = indexOfFeature(featureName);
		if (i == -1) { throw new DataStructureException("Feature does not exist (\"" + featureName + "\")"); }

		data.getFeatures().remove(i);
	}
	
	
	/**
	 * Adds a new state of the given feature.
	 * 
	 * @param featureName
	 * @param stateLabel
	 * @throws DataStructureException
	 */
	public void addState(String featureName, String stateLabel) throws DataStructureException {
		int i = indexOfFeature(featureName);
		if (i == -1) { throw new DataStructureException("Feature does not exist (\"" + featureName + "\")"); }
		if (indexOfState(data.getFeatures().get(i), stateLabel) >= 0) { throw new DataStructureException("State already exists (\"" + stateLabel + "\")"); }
		
		StateData newSD = new StateData();
		newSD.setLabel(stateLabel);
		newSD.setConditionalProbabilities(new ArrayList<Double>());
		fillDoubles(newSD.getConditionalProbabilities(), 1d, data.getClassLabels().size());

		data.getFeatures().get(i).getStates().add(newSD);
	}
	
	
	/**
	 * Removes a state.
	 * 
	 * @param featureName
	 * @param stateLabel
	 * @throws DataStructureException
	 */
	public void removeState(String featureName, String stateLabel) throws DataStructureException {
		int i = indexOfFeature(featureName);
		if (i == -1) { throw new DataStructureException("Feature does not exist (\"" + featureName + "\")"); }

		int j = indexOfState(data.getFeatures().get(i), stateLabel);
		if (j == -1) { throw new DataStructureException("State does not exist (\"" + stateLabel + "\")"); }
		
		data.getFeatures().get(i).getStates().remove(j);
	}
	

	/**
	 * Set the prior probability of a class label.
	 * 
	 * @param classLabel
	 * @param prior
	 * @throws DataStructureException
	 */
	public void setPriorProbability(String classLabel, double priorProbability) throws DataStructureException {
		if (data == null) { throw new DataStructureException("Label does not exist (\"" + classLabel + "\")"); }

		int i = data.getClassLabels().indexOf(classLabel);
		if (i == -1) { throw new DataStructureException("Label does not exist (\"" + classLabel + "\")"); }

		data.getPriorProbabilities().set(i, priorProbability);
	}
	

	/**
	 * Set the conditional probability of a state given a class label.
	 * 
	 * @param ofState
	 * @param ofFeature
	 * @param givenLabel
	 * @param conditionalProbability
	 * @throws DataStructureException
	 */
	public void setConditionalProbability(String featureName, String ofState, String givenLabel, double conditionalProbability) throws DataStructureException {
		int i = indexOfFeature(featureName);
		if (i == -1) { throw new DataStructureException("Feature does not exist (\"" + featureName + "\")"); }

		int j = indexOfState(data.getFeatures().get(i), ofState);
		if (j == -1) { throw new DataStructureException("State does not exist (\"" + ofState + "\")"); }
		
		int k = data.getClassLabels().indexOf(givenLabel);
		if (k == -1) { throw new DataStructureException("Label does not exist (\"" + givenLabel + "\")"); }
		
		data.getFeatures().get(i).getStates().get(j).getConditionalProbabilities().set(k, conditionalProbability);
	}
	
	
	/**
	 * Returns all features.
	 * 
	 * @return
	 */
	public String[] getFeatures() {
		if (data == null) { initNaiveBayesClassifier(); }

		String[] res = new String[data.getFeatures().size()];
		for (int featureIndex = 0; featureIndex < res.length; featureIndex++) {
			res[featureIndex] = data.getFeatures().get(featureIndex).getName();
		}
		return res;
	}
	
	
	/**
	 * Returns the states of a feature.
	 * 
	 * @param featureName
	 * @return
	 * @throws DataStructureException
	 */
	public String[] getStates(String featureName) throws DataStructureException {
		int featureIndex = indexOfFeature(featureName);
		if (featureIndex == -1) { throw new DataStructureException("Feature does not exist (\"" + featureName + "\")"); }

		FeatureData fd = data.getFeatures().get(featureIndex);
		String[] res = new String[fd.getStates().size()];
		for (int stateIndex = 0; stateIndex < res.length; stateIndex++) {
			res[stateIndex] = fd.getStates().get(stateIndex).getLabel();
		}
		return res;		
	}
	
	
	/**
	 * Returns prior probabilities.
	 * 
	 * @return
	 */
	public Double[] getPriorProbabilities() {
		if (data == null) { initNaiveBayesClassifier(); }

		Double[] res = new Double[data.getPriorProbabilities().size()];
		return data.getPriorProbabilities().toArray(res);
	}
	
	
	/**
	 * Returns the conditional probabilities of some feature state given all class labels.
	 * 
	 * @param featureName
	 * @param ofState
	 * @return
	 * @throws DataStructureException
	 */
	public Double[] getConditionalProbabilities(String featureName, String ofState) throws DataStructureException {
		int i = indexOfFeature(featureName);
		if (i == -1) { throw new DataStructureException("Feature does not exist (\"" + featureName + "\")"); }

		FeatureData fData = data.getFeatures().get(i);
		int j = indexOfState(fData, ofState);
		if (j == -1) { throw new DataStructureException("State does not exist (\"" + ofState + "\")"); }

		StateData sData = fData.getStates().get(j);
		Double[] res = new Double[sData.getConditionalProbabilities().size()];
		return sData.getConditionalProbabilities().toArray(res);
	}
	
	
	/**
	 * Examines the consistency of prior and conditional probabilities and throws an exception if inconsistent.
	 * @throws DataStructureException
	 */
	public void validate() throws DataStructureException {
		double priorSum = 0d;
		for (double p : data.getPriorProbabilities()) {
			priorSum += p;
		}
		if (priorSum != 1.0d) {
			throw new DataStructureException("The sum of prior probabilities is " + priorSum + " (should be 1.0)");
		}
		
		for (FeatureData fd : data.getFeatures()) {
			String featureName = fd.getName();
			for (int classIndex = 0; classIndex < data.getClassLabels().size(); classIndex++) {
				String classLabel = data.getClassLabels().get(classIndex);
				double conditionalSum = 0d;
				for (int stateIndex = 0; stateIndex < fd.getStates().size(); stateIndex++) {
					conditionalSum += fd.getStates().get(stateIndex).getConditionalProbabilities().get(classIndex);
				}
				if (conditionalSum != 1.0d) {
					throw new DataStructureException("The sum of conditional probabilities of " + featureName + " given " +
														classLabel + " is " + conditionalSum + " (should be 1.0)");
				}
			}
		}
	}
	
	
	public Double[] classify(Map<String, String> observations) throws DataStructureException {
		validate();

		Double[] result = getPriorProbabilities();
		
		// Put observed features in array used in this algorithm:
		FeatureData[] observedFeatures = new FeatureData[observations.keySet().size()];
		StateData[] observedStates = new StateData[observations.keySet().size()];
		
		int obsIndex = 0;
		for (String observedFeature : observations.keySet()) {
			int featureIndex = indexOfFeature(observedFeature);
			if (featureIndex == -1) { throw new DataStructureException("Feature does not exist (\"" + observedFeature + "\")"); }
			
			int observedStateIndex = indexOfState(data.getFeatures().get(featureIndex), observations.get(observedFeature));
			if (observedStateIndex == -1) { throw new DataStructureException("State does not exist (\"" + observations.get(observedFeature) + "\")"); }
			
			observedFeatures[obsIndex] = data.getFeatures().get(featureIndex);
			observedStates[obsIndex] = observedFeatures[obsIndex].getStates().get(observedStateIndex);
			obsIndex++;
		}
		
		// Compute the scaling factor (Z):
		double evidenceScaling = 0;
		for (int classIndex = 0; classIndex < result.length; classIndex++) {
			double factor = 1;
			for (int featureIndex = 0; featureIndex < observedFeatures.length; featureIndex++) {
				factor *= observedStates[featureIndex].getConditionalProbabilities().get(classIndex);
			}
			evidenceScaling += factor * data.getPriorProbabilities().get(classIndex); 
		}
		
		// Compute the result:
		for (int classIndex = 0; classIndex < result.length; classIndex++) {
			double factor = 1;
			for (int featureIndex = 0; featureIndex < observedFeatures.length; featureIndex++) {
				factor *= observedStates[featureIndex].getConditionalProbabilities().get(classIndex);
			}
			result[classIndex] *= factor / evidenceScaling;
		}
		
		return result;
	}
	
	
	/**
	 * This will create the underlying structures.
	 */
	private void initNaiveBayesClassifier() {
		data = new NaiveBayesData();
		data.setClassLabels(new ArrayList<String>());
		data.setPriorProbabilities(new ArrayList<Double>());
		data.setFeatures(new ArrayList<FeatureData>());
	}

	
	/**
	 * This methods finds the index of a feature.
	 * 
	 * @param featureName
	 * @return the index of the specified feature, or -1 if this list does not contain the feature.
	 */
	private int indexOfFeature(String featureName) {
		if (data == null) { return -1; }
		
		int i = 0;
		while (i < data.getFeatures().size()) {
			if (featureName.equals(data.getFeatures().get(i).getName())) {
				return i;
			}
			i++;
		}
		return -1;
	}
	
	
	/**
	 * This method finds the index of a state.
	 * @param feature
	 * @param stateLabel
	 * @return
	 */
	private int indexOfState(FeatureData feature, String stateLabel) {
		if (data == null) { return -1; }
		
		int i = 0;
		while (i < feature.getStates().size()) {
			if (stateLabel.equals(feature.getStates().get(i).getLabel())) {
				return i;
			}
			i++;
		}
		return -1;
	}

	
	private void fillDoubles(ArrayList<Double> a, double d, int count) {
		for (int i = 0; i < count; i++) {
			a.add(d);
		}
	}
}
