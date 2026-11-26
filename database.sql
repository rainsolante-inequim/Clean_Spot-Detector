CREATE DATABASE IF NOT EXISTS cleanspot;
USE cleanspot;

CREATE TABLE IF NOT EXISTS waste_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    item_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    disposal_method TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS waste_reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    waste_type VARCHAR(100) NOT NULL,
    location VARCHAR(255) NOT NULL,
    description TEXT,
    image VARCHAR(255),
    report_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'Reported'
);

INSERT INTO waste_items (item_name, category, disposal_method) VALUES
('Banana Peel', 'Biodegradable', 'Place in the biodegradable or compost bin.'),
('Food Scraps', 'Biodegradable', 'Place in the biodegradable or compost bin.'),
('Leaves', 'Biodegradable', 'Place in the biodegradable or compost bin.'),
('Plastic Bottle', 'Recyclable', 'Empty the bottle and place it in the recyclable bin.'),
('Cardboard', 'Recyclable', 'Keep it dry and place it in the recyclable bin.'),
('Paper', 'Recyclable', 'Place clean and dry paper in the recyclable bin.'),
('Glass Bottle', 'Recyclable', 'Handle carefully and place it in the appropriate recycling container.'),
('Used Tissue', 'Residual', 'Place in the residual waste bin.'),
('Candy Wrapper', 'Residual', 'Place in the residual waste bin.'),
('Battery', 'Special Waste', 'Bring it to an appropriate special-waste collection point.'),
('Old Phone', 'Special Waste', 'Bring electronic waste to an appropriate e-waste collection point.');
