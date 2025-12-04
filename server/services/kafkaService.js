/**
 * Kafka Service for storing feedback data for model retraining
 * TODO: Connect to actual Kafka cluster
 * For now, stores feedback locally in JSON file
 */

const fs = require('fs');
const path = require('path');

const FEEDBACK_FILE = path.join(__dirname, '../../feedback_data.json');

// Initialize feedback file if it doesn't exist
if (!fs.existsSync(FEEDBACK_FILE)) {
  fs.writeFileSync(FEEDBACK_FILE, JSON.stringify({ feedback: [] }, null, 2));
}

/**
 * Store feedback for incorrect prediction
 * @param {Object} feedbackData - The feedback data
 * @param {Array} feedbackData.logs - User's activity logs
 * @param {Object} feedbackData.prediction - Model's prediction
 * @param {string} feedbackData.actualApp - What the user actually accessed
 * @param {string} feedbackData.userId - User identifier
 * @param {string} feedbackData.timestamp - When feedback was submitted
 */
exports.storeFeedback = async (feedbackData) => {
  try {
    // TODO: Replace with actual Kafka producer
    // const kafka = new Kafka({ brokers: ['localhost:9092'] });
    // const producer = kafka.producer();
    // await producer.connect();
    // await producer.send({
    //   topic: 'model-feedback',
    //   messages: [{ value: JSON.stringify(feedbackData) }]
    // });
    // await producer.disconnect();

    // For now, append to local JSON file
    const data = JSON.parse(fs.readFileSync(FEEDBACK_FILE, 'utf8'));
    data.feedback.push({
      ...feedbackData,
      id: `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date().toISOString()
    });
    
    fs.writeFileSync(FEEDBACK_FILE, JSON.stringify(data, null, 2));
    
    console.log('✅ Feedback stored successfully for retraining');
    console.log(`   Predicted: ${feedbackData.prediction.next_pattern}`);
    console.log(`   Actual: ${feedbackData.actualApp}`);
    console.log(`   Total feedback entries: ${data.feedback.length}`);
    
    return { success: true, feedbackId: data.feedback[data.feedback.length - 1].id };
    
  } catch (error) {
    console.error('❌ Error storing feedback:', error);
    throw error;
  }
};

/**
 * Get all feedback entries (for admin/debugging)
 */
exports.getAllFeedback = () => {
  try {
    const data = JSON.parse(fs.readFileSync(FEEDBACK_FILE, 'utf8'));
    return data.feedback;
  } catch (error) {
    console.error('Error reading feedback:', error);
    return [];
  }
};

/**
 * Get feedback statistics
 */
exports.getFeedbackStats = () => {
  try {
    const feedback = exports.getAllFeedback();
    const stats = {
      total: feedback.length,
      byPattern: {},
      recent: feedback.slice(-10)
    };
    
    feedback.forEach(f => {
      const pattern = f.prediction?.next_pattern || 'unknown';
      stats.byPattern[pattern] = (stats.byPattern[pattern] || 0) + 1;
    });
    
    return stats;
  } catch (error) {
    console.error('Error calculating stats:', error);
    return { total: 0, byPattern: {}, recent: [] };
  }
};
