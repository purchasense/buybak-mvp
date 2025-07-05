import React, { useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Avatar,
    Button,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Divider,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    Chip,
    Grid,
    IconButton
} from '@mui/material';
import {
    Person as PersonIcon,
    Email as EmailIcon,
    Phone as PhoneIcon,
    LocationOn as LocationIcon,
    Edit as EditIcon,
    CameraAlt as CameraIcon,
    Verified as VerifiedIcon,
    Star as StarIcon,
    TrendingUp as TrendingUpIcon,
    AccountBalance as AccountBalanceIcon,
    CalendarToday as CalendarIcon,
    Language as LanguageIcon,
    Business as BusinessIcon
} from '@mui/icons-material';

const MobileProfile = () => {
    const [editDialogOpen, setEditDialogOpen] = useState(false);
    const [profileData, setProfileData] = useState({
        name: 'John Doe',
        email: 'john.doe@example.com',
        phone: '+1 (555) 123-4567',
        location: 'New York, NY',
        company: 'Wine Investment LLC',
        memberSince: 'January 2024',
        verified: true,
        investmentLevel: 'Premium',
        totalInvestments: 15,
        portfolioValue: '$125,000',
        languages: ['English', 'French', 'Spanish']
    });

    const handleSaveProfile = (newData) => {
        setProfileData({ ...profileData, ...newData });
        setEditDialogOpen(false);
    };

    const profileStats = [
        {
            icon: <TrendingUpIcon sx={{ color: '#27ae60' }} />,
            label: 'Total Investments',
            value: profileData.totalInvestments,
            color: '#27ae60'
        },
        {
            icon: <AccountBalanceIcon sx={{ color: '#3498db' }} />,
            label: 'Portfolio Value',
            value: profileData.portfolioValue,
            color: '#3498db'
        },
        {
            icon: <StarIcon sx={{ color: '#f39c12' }} />,
            label: 'Investment Level',
            value: profileData.investmentLevel,
            color: '#f39c12'
        }
    ];

    const profileSections = [
        {
            title: 'Personal Information',
            icon: <PersonIcon sx={{ color: '#3498db' }} />,
            items: [
                {
                    primary: 'Full Name',
                    secondary: profileData.name,
                    icon: <PersonIcon />,
                    editable: true
                },
                {
                    primary: 'Email Address',
                    secondary: profileData.email,
                    icon: <EmailIcon />,
                    editable: true
                },
                {
                    primary: 'Phone Number',
                    secondary: profileData.phone,
                    icon: <PhoneIcon />,
                    editable: true
                },
                {
                    primary: 'Location',
                    secondary: profileData.location,
                    icon: <LocationIcon />,
                    editable: true
                },
                {
                    primary: 'Company',
                    secondary: profileData.company,
                    icon: <BusinessIcon />,
                    editable: true
                }
            ]
        },
        {
            title: 'Account Details',
            icon: <VerifiedIcon sx={{ color: '#27ae60' }} />,
            items: [
                {
                    primary: 'Member Since',
                    secondary: profileData.memberSince,
                    icon: <CalendarIcon />,
                    editable: false
                },
                {
                    primary: 'Verification Status',
                    secondary: profileData.verified ? 'Verified' : 'Pending',
                    icon: <VerifiedIcon />,
                    editable: false,
                    chip: profileData.verified ? 
                        <Chip label="Verified" size="small" sx={{ backgroundColor: '#27ae60', color: 'white' }} /> : 
                        <Chip label="Pending" size="small" sx={{ backgroundColor: '#f39c12', color: 'white' }} />
                },
                {
                    primary: 'Languages',
                    secondary: profileData.languages.join(', '),
                    icon: <LanguageIcon />,
                    editable: true
                }
            ]
        }
    ];

    return (
        <Box sx={{ 
            backgroundColor: '#f8f9fa',
            minHeight: '100vh',
            pb: 8
        }}>
            {/* Profile Header */}
            <Card sx={{ 
                mb: 2, 
                backgroundColor: '#2c3e50',
                color: 'white',
                borderRadius: 0
            }}>
                <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                        <Box sx={{ position: 'relative' }}>
                            <Avatar 
                                sx={{ 
                                    width: 80, 
                                    height: 80,
                                    border: '3px solid #3498db'
                                }}
                            >
                                <PersonIcon sx={{ fontSize: 40 }} />
                            </Avatar>
                            <IconButton 
                                sx={{ 
                                    position: 'absolute',
                                    bottom: 0,
                                    right: 0,
                                    backgroundColor: '#3498db',
                                    color: 'white',
                                    width: 30,
                                    height: 30,
                                    '&:hover': {
                                        backgroundColor: '#2980b9'
                                    }
                                }}
                            >
                                <CameraIcon sx={{ fontSize: 16 }} />
                            </IconButton>
                        </Box>
                        <Box sx={{ flex: 1 }}>
                            <Typography variant="h5" sx={{ 
                                fontWeight: 700,
                                color: 'white',
                                mb: 1
                            }}>
                                {profileData.name}
                            </Typography>
                            <Typography variant="body2" sx={{ 
                                color: '#bdc3c7',
                                mb: 1
                            }}>
                                {profileData.email}
                            </Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Chip 
                                    label={profileData.investmentLevel} 
                                    size="small" 
                                    sx={{ 
                                        backgroundColor: '#f39c12', 
                                        color: 'white',
                                        fontWeight: 600
                                    }} 
                                />
                                {profileData.verified && (
                                    <Chip 
                                        label="Verified" 
                                        size="small" 
                                        sx={{ 
                                            backgroundColor: '#27ae60', 
                                            color: 'white',
                                            fontWeight: 600
                                        }} 
                                    />
                                )}
                            </Box>
                        </Box>
                        <IconButton 
                            onClick={() => setEditDialogOpen(true)}
                            sx={{ 
                                backgroundColor: 'rgba(255,255,255,0.1)',
                                color: 'white',
                                '&:hover': {
                                    backgroundColor: 'rgba(255,255,255,0.2)'
                                }
                            }}
                        >
                            <EditIcon />
                        </IconButton>
                    </Box>
                </CardContent>
            </Card>

            {/* Profile Stats */}
            <Box sx={{ px: 2, mb: 2 }}>
                <Grid container spacing={2}>
                    {profileStats.map((stat, index) => (
                        <Grid item xs={4} key={index}>
                            <Card sx={{ 
                                textAlign: 'center',
                                borderRadius: 2,
                                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                            }}>
                                <CardContent sx={{ p: 2 }}>
                                    <Box sx={{ mb: 1 }}>
                                        {stat.icon}
                                    </Box>
                                    <Typography variant="h6" sx={{ 
                                        fontWeight: 700,
                                        color: stat.color,
                                        mb: 0.5
                                    }}>
                                        {stat.value}
                                    </Typography>
                                    <Typography variant="caption" sx={{ 
                                        color: '#7f8c8d',
                                        fontWeight: 500
                                    }}>
                                        {stat.label}
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            </Box>

            {/* Profile Sections */}
            <Box sx={{ px: 2 }}>
                {profileSections.map((section, sectionIndex) => (
                    <Card key={sectionIndex} sx={{ 
                        mb: 2,
                        borderRadius: 2,
                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                    }}>
                        <CardContent sx={{ p: 0 }}>
                            {/* Section Header */}
                            <Box sx={{ 
                                p: 2, 
                                backgroundColor: '#f8f9fa',
                                borderBottom: '1px solid #e9ecef'
                            }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                    {section.icon}
                                    <Typography variant="h6" sx={{ 
                                        fontWeight: 600,
                                        color: '#2c3e50'
                                    }}>
                                        {section.title}
                                    </Typography>
                                </Box>
                            </Box>

                            {/* Section Items */}
                            <List sx={{ p: 0 }}>
                                {section.items.map((item, itemIndex) => (
                                    <React.Fragment key={itemIndex}>
                                        <ListItem 
                                            sx={{ 
                                                cursor: item.editable ? 'pointer' : 'default',
                                                '&:hover': item.editable ? {
                                                    backgroundColor: '#f8f9fa'
                                                } : {}
                                            }}
                                            onClick={item.editable ? () => setEditDialogOpen(true) : undefined}
                                        >
                                            <ListItemIcon sx={{ color: '#7f8c8d' }}>
                                                {item.icon}
                                            </ListItemIcon>
                                            <ListItemText
                                                primary={
                                                    <Typography variant="body1" sx={{ 
                                                        fontWeight: 500,
                                                        color: '#2c3e50'
                                                    }}>
                                                        {item.primary}
                                                    </Typography>
                                                }
                                                secondary={
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
                                                        <Typography variant="body2" sx={{ 
                                                            color: '#7f8c8d'
                                                        }}>
                                                            {item.secondary}
                                                        </Typography>
                                                        {item.chip}
                                                    </Box>
                                                }
                                            />
                                            {item.editable && (
                                                <Typography variant="h6" sx={{ color: '#bdc3c7' }}>
                                                    ›
                                                </Typography>
                                            )}
                                        </ListItem>
                                        {itemIndex < section.items.length - 1 && (
                                            <Divider sx={{ ml: 4 }} />
                                        )}
                                    </React.Fragment>
                                ))}
                            </List>
                        </CardContent>
                    </Card>
                ))}
            </Box>

            {/* Edit Profile Dialog */}
            <Dialog 
                open={editDialogOpen} 
                onClose={() => setEditDialogOpen(false)}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle sx={{ 
                    color: '#2c3e50',
                    fontWeight: 600
                }}>
                    Edit Profile
                </DialogTitle>
                <DialogContent>
                    <Box sx={{ mt: 2 }}>
                        <TextField
                            fullWidth
                            label="Full Name"
                            variant="outlined"
                            defaultValue={profileData.name}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Email"
                            variant="outlined"
                            defaultValue={profileData.email}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Phone"
                            variant="outlined"
                            defaultValue={profileData.phone}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Location"
                            variant="outlined"
                            defaultValue={profileData.location}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Company"
                            variant="outlined"
                            defaultValue={profileData.company}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Languages (comma separated)"
                            variant="outlined"
                            defaultValue={profileData.languages.join(', ')}
                        />
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={() => setEditDialogOpen(false)}
                        sx={{ color: '#7f8c8d' }}
                    >
                        Cancel
                    </Button>
                    <Button 
                        onClick={() => handleSaveProfile({
                            name: 'John Doe',
                            email: 'john.doe@example.com',
                            phone: '+1 (555) 123-4567',
                            location: 'New York, NY',
                            company: 'Wine Investment LLC',
                            languages: ['English', 'French', 'Spanish']
                        })}
                        sx={{ 
                            color: '#3498db',
                            '&:hover': {
                                backgroundColor: '#ebf3fd'
                            }
                        }}
                    >
                        Save Changes
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default MobileProfile; 