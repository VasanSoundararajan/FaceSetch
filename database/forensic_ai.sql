--
-- Database: `forensic_ai`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL auto_increment,
  `username` varchar(50) default NULL,
  `password` varchar(50) default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`id`, `username`, `password`) VALUES
(1, 'admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `criminals`
--

CREATE TABLE `criminals` (
  `id` int(11) NOT NULL,
  `name` varchar(100) default NULL,
  `age` int(11) default NULL,
  `crime_year` int(11) default NULL,
  `crime_type` varchar(100) default NULL,
  `no_of_crimes` int(11) default NULL,
  `last_known_location` varchar(100) default NULL,
  `criminal_status` varchar(100) default NULL,
  `description` text,
  `prompt` text,
  `sketch_image` varchar(100) default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

--
-- Dumping data for table `criminals`
--


-- --------------------------------------------------------

--
-- Table structure for table `police`
--

CREATE TABLE `police` (
  `id` int(11) NOT NULL auto_increment,
  `name` varchar(100) default NULL,
  `email` varchar(100) default NULL,
  `mobile` varchar(20) default NULL,
  `location` varchar(100) default NULL,
  `station_name` varchar(100) default NULL,
  `police_id` varchar(50) default NULL,
  `username` varchar(50) default NULL,
  `password` varchar(50) default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=utf8 AUTO_INCREMENT=2 ;

--
-- Dumping data for table `police`
--

INSERT INTO `police` (`id`, `name`, `email`, `mobile`, `location`, `station_name`, `police_id`, `username`, `password`) VALUES
(1, 'Raj', 'akil@gmail.com', '8148956634', 'Trichy', 'trichy Station', 'P001', 'raj', '1234');
