package moltsen.AI.NaiveBayes;

/**
 * An exception used to inform about structural problems
 * in the Naive Bayes classifier.
 * 
 * @author  Lars Moltsen
 * @version 1.0
 */
public class DataStructureException extends Exception {

	public DataStructureException(String msg) {
		super(msg);
	}
}
