package moltsen.AI.NaiveBayes;

import java.util.HashMap;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import moltsen.AI.NaiveBayes.*;

/**
 * Unit test for simple App.
 */
public class NBTest 
    extends TestCase
{
	NaiveBayesClassifier c;
	
    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public NBTest( String testName )
    {
        super( testName );
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite()
    {
        return new TestSuite( NBTest.class );
    }
    

    /**
     * In all tests, we consider a diagnostic scenario where the algorithm should calculate
     * the probability of "Flu", "Measles", and "No disease" given the observation of
     * "Fever" ("yes" or "no") and "Red spots" ("yes" or "no").
     */
    protected void setUp() {
    	c = new NaiveBayesClassifier();
    	
    	try {
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

        	// Fully consistent model should validate:
    		c.validate();    		
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}
    }

    
    /**
     * The classifier should throw DataStructureException when attempting same
     * class label, feature name, or state name. 
     */
    public void testDoNotAllowDoublets()
    {
    	try {
    		c.addClassLabel("Flu");
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}

    	try {
    		c.addFeature("Fever");
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}

    	try {
    		c.addState("Fever", "yes");
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}
    }
    
    
    /**
     * Verify that structure changes as expected when deleting stuff.
     */
    public void testStructure() {
    	try {
       		c.removeLabel("Flu");

       		assertEquals("Prior probabilities array", 2, c.getPriorProbabilities().length);
       		assertEquals("Conditional probabilities array (Fever, yes)", 2, c.getConditionalProbabilities("Fever", "yes").length);
       		assertEquals("Conditional probabilities array (Red spots, no)", 2, c.getConditionalProbabilities("Red spots", "no").length);
       		
       		c.removeFeature("Fever");
       		
       		assertEquals("Features array", 1, c.getFeatures().length);
       		
       		c.removeState("Red spots", "no");
       		
       		assertEquals("States array", 1, c.getStates("Red spots").length);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}    	
    }
    
    
    /**
     * Verify that priors changes as expected when deleting stuff.
     */
    public void testPriorProbabilities() {
    	try {
       		c.removeLabel("Measles");

       		assertEquals("Prior of Flu", 0.06d, c.getPriorProbabilities()[0]);
       		assertEquals("Prior of No disease", 0.90d, c.getPriorProbabilities()[1]);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}    	
    }
    
    
    /**
     * Verify that conditionals changes as expected when deleting stuff.
     */
    public void testConditionalProbabilities() {
    	try {
       		c.removeLabel("Measles");

       		assertEquals("Conditional of Fever=yes given Flu", 0.90d, c.getConditionalProbabilities("Fever", "yes")[0]);
       		assertEquals("Conditional of Fever=yes given No disease", 0.01d, c.getConditionalProbabilities("Fever", "yes")[1]);

       		c.removeState("Fever", "yes");

       		assertEquals("Conditional of Fever=no given Flu", 0.10d, c.getConditionalProbabilities("Fever", "no")[0]);
       		assertEquals("Conditional of Fever=no given No disease", 0.99d, c.getConditionalProbabilities("Fever", "no")[1]);    	
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}

    	// Retrieving conditional probabilities of Fever=yes (removed above) should cause exception:
    	try {
    		Double[] x = c.getConditionalProbabilities("Fever", "yes");
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}
    }
    
    
    /**
     * The validation function is used to validate a model before classification. This
     * tests when we want it to raise an exception.
     */
    public void testValidation() {
    	// The starting point is valid:
    	try {
    		c.validate();
		}
		catch (Exception e) {
	        assertTrue("Exception should not happen: " + e.getMessage(), false);
		}

    	// Insert a crazy value:
    	try {
    		c.setConditionalProbability("Fever", "yes", "Measles", 1000.0d);
    		c.validate();
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}

    	// Removing Measles makes the priors inconsistent.
    	try {
       		c.removeLabel("Measles");
    		c.validate();
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}

    	// Fix the prior prob. issue...
    	try {
    		c.setPriorProbability("Flu", 0.10d);
    		c.validate();
		}
		catch (Exception e) {
	        assertTrue("Exception should not happen: " + e.getMessage(), false);
		}

    	// Removing state yes makes the conditionals inconsistent.
    	try {
    		c.removeState("Fever", "yes");
    		c.validate();
            assertTrue("Exception should happen", false);
    	}
    	catch (Exception e) {
    	}
    }
    
    
    /**
     * Test the classification algorithms.
     */
    public void testClassification() {
    	HashMap<String, String> observations = new HashMap<String, String>();
    	
    	try {
    		Double[] result = c.classify(observations);
       		assertEquals("Probability of Flu", 0.06d, result[0]);    	
       		assertEquals("Probability of Measles", 0.04d, result[1]);    	
       		assertEquals("Probability of No disease", 0.90d, result[2]);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}   	

    	observations.put("Red spots", "yes");
    	try {
    		Double[] result = c.classify(observations);

    		System.out.println("Results (Red spots=yes)");
    		System.out.println("Probability of Flu: " + result[0]);
    		System.out.println("Probability of Measles: " + result[1]);
    		System.out.println("Probability of No disease: " + result[2]);
    		
    		double sumOfProbs = 0;
    		for (int i = 0; i < result.length; i++) {
    			sumOfProbs += result[i];
    		}
       		assertEquals("The sum is 1.0", 1.0d, sumOfProbs, 0.0000001);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}   	

    	observations.put("Fever", "yes");
    	observations.put("Red spots", "no");
    	try {
    		Double[] result = c.classify(observations);
    		System.out.println("Results (Fever=yes, Red spots=no)");
    		System.out.println("Probability of Flu: " + result[0]);
    		System.out.println("Probability of Measles: " + result[1]);
    		System.out.println("Probability of No disease: " + result[2]);
    		
    		double sumOfProbs = 0;
    		for (int i = 0; i < result.length; i++) {
    			sumOfProbs += result[i];
    		}
       		assertEquals("The sum is 1.0", 1.0d, sumOfProbs, 0.0000001);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}   	
    }
    
    
    /**
     * Make sure the algorithm works when conditionals are a bit extreme.
     */
    public void testClassificationExtremes() {
    	HashMap<String, String> observations = new HashMap<String, String>();
    	
    	// Injecting a special "Flu detector" feature:
    	try {
    		c.addFeature("Flu detector");
    		c.addState("Flu detector", "on");
    		c.addState("Flu detector", "off");

    		c.setConditionalProbability("Flu detector", "on", "Flu", 0.99d);
    		c.setConditionalProbability("Flu detector", "on", "Measles", 0.00d); // Flu detector cannot trigger when measles...
    		c.setConditionalProbability("Flu detector", "on", "No disease", 0.00d); // Flu detector cannot trigger when no disease...
    		
    		c.setConditionalProbability("Flu detector", "off", "Flu", 0.01d);
    		c.setConditionalProbability("Flu detector", "off", "Measles", 1.0d);
    		c.setConditionalProbability("Flu detector", "off", "No disease", 1.0d);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}   	

    	// Flu is certain:
    	observations.put("Flu detector", "on");
    	try {
    		Double[] result = c.classify(observations);

    		System.out.println("Results (Flu detector=on)");
    		System.out.println("Probability of Flu: " + result[0]);
    		System.out.println("Probability of Measles: " + result[1]);
    		System.out.println("Probability of No disease: " + result[2]);

       		assertEquals("Probability of Flu", 1.0d, result[0], 0.0000001);
       		assertEquals("Probability of Measles", 0.0d, result[1], 0.0000001);
       		assertEquals("Probability of No disease", 0.0d, result[2], 0.0000001);

       		double sumOfProbs = 0;
    		for (int i = 0; i < result.length; i++) {
    			sumOfProbs += result[i];
    		}
       		assertEquals("The sum is 1.0", 1.0d, sumOfProbs, 0.000001);
    	}
    	catch (Exception e) {
            assertTrue("Exception should not happen: " + e.getMessage(), false);
    	}   	
    }
}
