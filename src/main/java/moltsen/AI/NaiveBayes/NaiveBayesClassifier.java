package moltsen.AI.NaiveBayes;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import moltsen.AI.NaiveBayes.model.*;

/**
 * <p>Add class labels, features with any number of states, prior and conditional
 * probabilities. Then compute the probability of each class labels given observed
 * features.</p>
 *
 * <h2>Example</h2>
 * 
 * <p>Imagine you want to use the classifier to quickly diagnose if a patient
 * has either the flu or measles, and you are able to observe fever and (red)
 * spots on the skin of the patient. This can be modelled as follows:</p>
 * 
 * <h3>Class labels:</h3>
 * <ul>
 * <li>Flu</li>
 * <li>Measles</li>
 * <li>No disease</li>
 * </ul>
 * 
 * <h3>Features:</h3>
 * <ul>
 * <li>Fever (yes, no)</li>
 * <li>Red spots (yes, no)</li>
 * </ul>
 * 
 * <p>This can be programmed and tested as follows.</p>
 * 
 * <pre>
try {
	NaiveBayesClassifier c = new NaiveBayesClassifier();
	c.addClassLabel("Flu");
	c.addClassLabel("Measles");
	c.addClassLabel("No disease");

	c.addFeature("Fever");
	c.addState("Fever", "yes");
	c.addState("Fever", "no");

	c.addFeature("Red spots");
	c.addState("Red spots", "yes");
	c.addState("Red spots", "no");

	c.setPriorProbability("Flu", 0.06d);
	c.setPriorProbability("Measles", 0.04d);
	c.setPriorProbability("No disease", 0.90d);
    
	// Probability of observing Fever=yes given different diseases 
	c.setConditionalProbability("Fever", "yes", "Flu", 0.90d);
	c.setConditionalProbability("Fever", "yes", "Measles", 0.90d);
	c.setConditionalProbability("Fever", "yes", "No disease", 0.01d);
    	
	// Probability of observing Fever=no given different diseases 
	c.setConditionalProbability("Fever", "no", "Flu", 0.10d);
	c.setConditionalProbability("Fever", "no", "Measles", 0.10d);
	c.setConditionalProbability("Fever", "no", "No disease", 0.99d);

	// Probability of observing Red spots=yes given different diseases
	c.setConditionalProbability("Red spots", "yes", "Flu", 0.05d);
	c.setConditionalProbability("Red spots", "yes", "Measles", 0.90d);
	c.setConditionalProbability("Red spots", "yes", "No disease", 0.01d);
    
	// Probability of observing Red spots=no given different diseases 
	c.setConditionalProbability("Red spots", "no", "Flu", 0.95d);
	c.setConditionalProbability("Red spots", "no", "Measles", 0.10d);
	c.setConditionalProbability("Red spots", "no", "No disease", 0.99d);

	// Inject observations as a Map object
	HashMap<String, String> observations = new HashMap<String, String>();
	observations.put("Fever", "yes");
	observations.put("Red spots", "no");
	Double[] result = c.classify(observations);
	
	// Present results
	System.out.println("Results (Fever=yes, Red spots=no)");
	System.out.println("Probability of Flu: " + result[0]);
	System.out.println("Probability of Measles: " + result[1]);
	System.out.println("Probability of No disease: " + result[2]);
catch (Exception e) {
	// Fix something...
}
 * </pre>
 *
 * <p>Output from this code:</p>
 * 
 * <pre>
  Results (Fever=yes, Red spots=no)
  Probability of Flu: 0.8039492242595203
  Probability of Measles: 0.056417489421720736
  Probability of No disease: 0.13963328631875882
 * </pre>
 * 
 * @author  Lars Moltsen
 * @version 1.0
 */
public class NaiveBayesClassifier {
	private NaiveBayesData data;

	
	/**
	 * Add a class label.
	 * 
	 * @param label The new class label
	 * @throws DataStructureException 
	 */
	public void addClassLabel(String label) throws DataStructureException {
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
	 * @param label The target class label
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
	 * @param name The name of the new feature
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
	 * @param name The name of the target feature.
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
	 * @param featureName The name of the target feature.
	 * @param stateLabel The new state label.
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
	 * @param featureName The name of the target feature.
	 * @param stateLabel The target state label.
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
	 * Set the prior (default) probability of a class label. The prior probability is
	 * what you get if you use the classify method with no observed features. 
	 * 
	 * @param classLabel The name of the target class label.
	 * @param prior The prior probability of the class label.
	 * @throws DataStructureException
	 */
	public void setPriorProbability(String classLabel, double priorProbability) throws DataStructureException {
		if (data == null) { throw new DataStructureException("Label does not exist (\"" + classLabel + "\")"); }

		int i = data.getClassLabels().indexOf(classLabel);
		if (i == -1) { throw new DataStructureException("Label does not exist (\"" + classLabel + "\")"); }

		data.getPriorProbabilities().set(i, priorProbability);
	}
	

	/**
	 * Set the conditional probability of a state given a class label. The conditional
	 * probability is what you would expect for this feature if the given class label
	 * was known.
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
	 * Returns all class labels.
	 * 
	 * @return Class labels as an array of String.
	 */
	public String[] getClassLabels() {
		if (data == null) { initNaiveBayesClassifier(); }
		
		String[] result = new String[data.getClassLabels().size()];
		return data.getClassLabels().toArray(result);
	}
	
	/**
	 * Returns all features.
	 * 
	 * @return Features as an array of String.
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
	 * @param featureName The feature.
	 * @return States of a feature as an array of String.
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
	 * @return The prior probability distribution over the class labels.
	 */
	public Double[] getPriorProbabilities() {
		if (data == null) { initNaiveBayesClassifier(); }

		Double[] res = new Double[data.getPriorProbabilities().size()];
		return data.getPriorProbabilities().toArray(res);
	}
	
	
	/**
	 * Returns the conditional probabilities of some feature state given all class labels.
	 * 
	 * @param featureName The feature of interest.
	 * @param ofState The state of interest.
	 * @return The conditional probability of the given state per class label.
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
	 * Examines the consistency of prior and conditional probabilities and
	 * throws an exception if inconsistent. This is done as the first step
	 * in the classify method.
	 * 
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
	
	
	/**
	 * The Naive Bayes classifiaction algorithm. The structure of the classifier
	 * must be complete and consistent. If not, an exception will be thrown.
	 * 
	 * @param observations A map (e.g. HashMap) of feature (key) and state (value) pairs. 
	 * @return A probability distribution over the class labels given the observations.
	 * @throws DataStructureException
	 */
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
