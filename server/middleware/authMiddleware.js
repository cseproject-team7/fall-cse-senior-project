const jwt = require('jsonwebtoken');

const authMiddleware = (req, res, next) => {
    try {
        // 1. Get token from "Authorization: Bearer <token>"
        const authHeader = req.headers.authorization;
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            return res.status(401).json({ success: false, message: 'Not authorized, no token' });
        }

        const token = authHeader.split(' ')[1]; // Get just the token

        // 2. Verify the token (read JWT_SECRET dynamically for testing)
        const decoded = jwt.verify(token, process.env.JWT_SECRET);

        // 3. Token is valid. Add user data to the request.
        req.user = { userId: decoded.userId, email: decoded.email };

        // 4. Continue to the protected route
        next();
    } catch (error) {
        console.error('Auth middleware error:', error.message);
        
        // Provide specific error message for expired tokens
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({ success: false, message: 'Token expired, please login again' });
        }
        
        res.status(401).json({ success: false, message: 'Not authorized, token failed' });
    }
};

module.exports = authMiddleware;