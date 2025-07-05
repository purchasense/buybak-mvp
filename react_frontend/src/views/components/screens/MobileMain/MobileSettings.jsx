import React, { useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Switch,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Divider,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    FormControlLabel,
    Chip,
    Avatar
} from '@mui/material';
import {
    Settings as SettingsIcon,
    Notifications as NotificationsIcon,
    Security as SecurityIcon,
    Palette as PaletteIcon,
    Help as HelpIcon,
    Info as InfoIcon,
    Person as PersonIcon,
    Email as EmailIcon,
    Phone as PhoneIcon,
    Language as LanguageIcon,
    Brightness4 as DarkModeIcon,
    Brightness7 as LightModeIcon,
    VolumeUp as VolumeIcon,
    Delete as DeleteIcon,
    Logout as LogoutIcon
} from '@mui/icons-material';

const MobileSettings = () => {
    const [notifications, setNotifications] = useState(true);
    const [darkMode, setDarkMode] = useState(false);
    const [emailNotifications, setEmailNotifications] = useState(true);
    const [pushNotifications, setPushNotifications] = useState(true);
    const [language, setLanguage] = useState('English');
    const [volume, setVolume] = useState(80);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [profileDialogOpen, setProfileDialogOpen] = useState(false);

    const handleDeleteAccount = () => {
        setDeleteDialogOpen(false);
        // Handle account deletion logic
    };

    const handleLogout = () => {
        // Handle logout logic
    };

    const settingsSections = [
        {
            title: 'Account',
            icon: <PersonIcon sx={{ color: '#3498db' }} />,
            items: [
                {
                    primary: 'Profile Information',
                    secondary: 'Update your personal details',
                    action: () => setProfileDialogOpen(true),
                    icon: <PersonIcon />
                },
                {
                    primary: 'Email Address',
                    secondary: 'john.doe@example.com',
                    action: () => {},
                    icon: <EmailIcon />
                },
                {
                    primary: 'Phone Number',
                    secondary: '+1 (555) 123-4567',
                    action: () => {},
                    icon: <PhoneIcon />
                }
            ]
        },
        {
            title: 'Notifications',
            icon: <NotificationsIcon sx={{ color: '#e74c3c' }} />,
            items: [
                {
                    primary: 'Push Notifications',
                    secondary: 'Receive alerts on your device',
                    action: () => setPushNotifications(!pushNotifications),
                    switch: true,
                    switchValue: pushNotifications,
                    icon: <NotificationsIcon />
                },
                {
                    primary: 'Email Notifications',
                    secondary: 'Get updates via email',
                    action: () => setEmailNotifications(!emailNotifications),
                    switch: true,
                    switchValue: emailNotifications,
                    icon: <EmailIcon />
                },
                {
                    primary: 'Market Alerts',
                    secondary: 'Price change notifications',
                    action: () => setNotifications(!notifications),
                    switch: true,
                    switchValue: notifications,
                    icon: <NotificationsIcon />
                }
            ]
        },
        {
            title: 'Appearance',
            icon: <PaletteIcon sx={{ color: '#f39c12' }} />,
            items: [
                {
                    primary: 'Dark Mode',
                    secondary: 'Switch between light and dark themes',
                    action: () => setDarkMode(!darkMode),
                    switch: true,
                    switchValue: darkMode,
                    icon: darkMode ? <DarkModeIcon /> : <LightModeIcon />
                },
                {
                    primary: 'Language',
                    secondary: language,
                    action: () => {},
                    icon: <LanguageIcon />
                }
            ]
        },
        {
            title: 'Privacy & Security',
            icon: <SecurityIcon sx={{ color: '#27ae60' }} />,
            items: [
                {
                    primary: 'Two-Factor Authentication',
                    secondary: 'Enhanced account security',
                    action: () => {},
                    icon: <SecurityIcon />
                },
                {
                    primary: 'Data Privacy',
                    secondary: 'Manage your data preferences',
                    action: () => {},
                    icon: <SecurityIcon />
                }
            ]
        },
        {
            title: 'Support',
            icon: <HelpIcon sx={{ color: '#9b59b6' }} />,
            items: [
                {
                    primary: 'Help Center',
                    secondary: 'Find answers to common questions',
                    action: () => {},
                    icon: <HelpIcon />
                },
                {
                    primary: 'Contact Support',
                    secondary: 'Get in touch with our team',
                    action: () => {},
                    icon: <HelpIcon />
                },
                {
                    primary: 'About BuyBak',
                    secondary: 'Version 1.0.0',
                    action: () => {},
                    icon: <InfoIcon />
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
            {/* Header */}
            <Card sx={{ 
                mb: 2, 
                backgroundColor: '#2c3e50',
                color: 'white',
                borderRadius: 0
            }}>
                <CardContent sx={{ p: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Avatar sx={{ 
                            backgroundColor: '#3498db',
                            width: 50,
                            height: 50
                        }}>
                            <SettingsIcon />
                        </Avatar>
                        <Box>
                            <Typography variant="h5" sx={{ 
                                fontWeight: 700,
                                color: 'white'
                            }}>
                                Settings
                            </Typography>
                            <Typography variant="body2" sx={{ 
                                color: '#bdc3c7'
                            }}>
                                Manage your preferences
                            </Typography>
                        </Box>
                    </Box>
                </CardContent>
            </Card>

            {/* Settings Sections */}
            <Box sx={{ px: 2 }}>
                {settingsSections.map((section, sectionIndex) => (
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
                                                cursor: 'pointer',
                                                '&:hover': {
                                                    backgroundColor: '#f8f9fa'
                                                }
                                            }}
                                            onClick={item.action}
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
                                                    <Typography variant="body2" sx={{ 
                                                        color: '#7f8c8d'
                                                    }}>
                                                        {item.secondary}
                                                    </Typography>
                                                }
                                            />
                                            {item.switch ? (
                                                <Switch
                                                    checked={item.switchValue}
                                                    onChange={item.action}
                                                    sx={{
                                                        '& .MuiSwitch-switchBase.Mui-checked': {
                                                            color: '#3498db',
                                                        },
                                                        '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                                                            backgroundColor: '#3498db',
                                                        },
                                                    }}
                                                />
                                            ) : (
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

                {/* Danger Zone */}
                <Card sx={{ 
                    mb: 2,
                    borderRadius: 2,
                    border: '1px solid #e74c3c',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                }}>
                    <CardContent sx={{ p: 0 }}>
                        <Box sx={{ 
                            p: 2, 
                            backgroundColor: '#fdf2f2',
                            borderBottom: '1px solid #e74c3c'
                        }}>
                            <Typography variant="h6" sx={{ 
                                fontWeight: 600,
                                color: '#e74c3c'
                            }}>
                                ⚠️ Danger Zone
                            </Typography>
                        </Box>
                        <List sx={{ p: 0 }}>
                            <ListItem 
                                sx={{ 
                                    cursor: 'pointer',
                                    '&:hover': {
                                        backgroundColor: '#fdf2f2'
                                    }
                                }}
                                onClick={() => setDeleteDialogOpen(true)}
                            >
                                <ListItemIcon sx={{ color: '#e74c3c' }}>
                                    <DeleteIcon />
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Typography variant="body1" sx={{ 
                                            fontWeight: 500,
                                            color: '#e74c3c'
                                        }}>
                                            Delete Account
                                        </Typography>
                                    }
                                    secondary="Permanently delete your account and all data"
                                />
                                <Typography variant="h6" sx={{ color: '#e74c3c' }}>
                                    ›
                                </Typography>
                            </ListItem>
                            <Divider />
                            <ListItem 
                                sx={{ 
                                    cursor: 'pointer',
                                    '&:hover': {
                                        backgroundColor: '#fdf2f2'
                                    }
                                }}
                                onClick={handleLogout}
                            >
                                <ListItemIcon sx={{ color: '#e74c3c' }}>
                                    <LogoutIcon />
                                </ListItemIcon>
                                <ListItemText
                                    primary={
                                        <Typography variant="body1" sx={{ 
                                            fontWeight: 500,
                                            color: '#e74c3c'
                                        }}>
                                            Logout
                                        </Typography>
                                    }
                                    secondary="Sign out of your account"
                                />
                                <Typography variant="h6" sx={{ color: '#e74c3c' }}>
                                    ›
                                </Typography>
                            </ListItem>
                        </List>
                    </CardContent>
                </Card>
            </Box>

            {/* Delete Account Dialog */}
            <Dialog 
                open={deleteDialogOpen} 
                onClose={() => setDeleteDialogOpen(false)}
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle sx={{ 
                    color: '#e74c3c',
                    fontWeight: 600
                }}>
                    Delete Account
                </DialogTitle>
                <DialogContent>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                        Are you sure you want to delete your account? This action cannot be undone and will permanently remove all your data, including:
                    </Typography>
                    <Box sx={{ ml: 2 }}>
                        <Typography variant="body2" sx={{ mb: 1 }}>• Your profile information</Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>• Investment portfolio</Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>• Transaction history</Typography>
                        <Typography variant="body2" sx={{ mb: 1 }}>• Saved preferences</Typography>
                    </Box>
                    <TextField
                        fullWidth
                        label="Type 'DELETE' to confirm"
                        variant="outlined"
                        sx={{ mt: 2 }}
                    />
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={() => setDeleteDialogOpen(false)}
                        sx={{ color: '#7f8c8d' }}
                    >
                        Cancel
                    </Button>
                    <Button 
                        onClick={handleDeleteAccount}
                        sx={{ 
                            color: '#e74c3c',
                            '&:hover': {
                                backgroundColor: '#fdf2f2'
                            }
                        }}
                    >
                        Delete Account
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Profile Dialog */}
            <Dialog 
                open={profileDialogOpen} 
                onClose={() => setProfileDialogOpen(false)}
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
                            defaultValue="John Doe"
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Email"
                            variant="outlined"
                            defaultValue="john.doe@example.com"
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Phone"
                            variant="outlined"
                            defaultValue="+1 (555) 123-4567"
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Location"
                            variant="outlined"
                            defaultValue="New York, NY"
                        />
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button 
                        onClick={() => setProfileDialogOpen(false)}
                        sx={{ color: '#7f8c8d' }}
                    >
                        Cancel
                    </Button>
                    <Button 
                        onClick={() => setProfileDialogOpen(false)}
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

export default MobileSettings; 