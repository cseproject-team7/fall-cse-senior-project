const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');

exports.login = async (req, res) => {
    try {
        const { email, password } = req.body;

        // Get credentials from environment (read dynamically for testing)
        const ADMIN_USER = process.env.ADMIN_USER;
        const ADMIN_HASH = process.env.ADMIN_HASH;
        const JWT_SECRET = process.env.JWT_SECRET;

        // 1. Check for email and password
        if (!email || !password) {
            return res.status(400).json({ success: false, message: 'Please provide email and password' });
        }

        // 2. Check if email matches
        if (email.toLowerCase() !== ADMIN_USER) {
            return res.status(401).json({ success: false, message: 'Invalid credentials' });
        }

        // 3. Securely compare password with stored hash
        const isMatch = bcrypt.compareSync(password, ADMIN_HASH);

        if (isMatch) {
            // 4. Passwords match! Create a token.
            const token = jwt.sign(
                { userId: 'admin', email: ADMIN_USER }, 
                JWT_SECRET, 
                { expiresIn: '24h' } // Token lasts for 24 hours
            );
            // 5. Send the token to the frontend
            res.json({ success: true, message: 'Login successful', token: token });
        } else {
            // 6. Wrong password
            res.status(401).json({ success: false, message: 'Invalid credentials' });
        }
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ success: false, message: 'Server error during login' });
    }
};