import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class RecommendationSystem {
    public static void main(String[] args) {
        try {
            // Load data from a file (userID, itemID, preference)
            DataModel model = new FileDataModel(new File("data.csv"));

            // Define similarity algorithm
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // Define user neighborhood
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            // Create recommender
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // Get recommendations for a user (e.g., user ID 1)
            List<RecommendedItem> recommendations = recommender.recommend(1, 3);

            // Print recommendations
            System.out.println("Recommended Items for User 1:");
            for (RecommendedItem recommendation : recommendations) {
                System.out.println(recommendation.getItemID() + " - Score: " + recommendation.getValue());
            }

            // Evaluate recommender performance
            RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
            double evaluationScore = evaluator.evaluate(recommender, null, model, 0.7, 1.0);
            System.out.println("Evaluation Score: " + evaluationScore);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
