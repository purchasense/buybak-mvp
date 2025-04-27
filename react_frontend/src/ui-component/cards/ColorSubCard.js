import PropTypes from 'prop-types';
import React from 'react';

import { Card, CardContent, CardHeader, Divider, Typography } from '@mui/material';

// ===========================|| CUSTOM SUB CARD ||=========================== //

const ColorSubCard = ({ children, content, contentClass, darkTitle, secondary, border, background, sx = {}, title, ...others }) => {

    return (
        <Card
            sx={{
                border: '1px solid',
                borderColor: border,
                background: background,
                ':hover': {
                    boxShadow: '0 2px 14px 0 rgb(33 150 243 / 10%)'  // '0 2px 14px 0 rgb(32 40 45 / 10%)'
                },
                ...sx
            }}
            {...others}
        >
            {/* card header and action */}
            {!darkTitle && title && <CardHeader sx={{ p: 2.5 }} title={title} action={secondary} />}
            {darkTitle && title && <CardHeader sx={{ p: 2.5 }} title={title} action={secondary} />}

            {/* content & header divider */}
            {title && (
                <Divider
                    sx={{
                        opacity: 1,
                        borderColor: border,
                    }}
                />
            )}

            {/* card content */}
            {content && (
                <CardContent sx={{ p: 2.5 }} className={contentClass}>
                    {children}
                </CardContent>
            )}
            {!content && children}
        </Card>
    );
};

ColorSubCard.propTypes = {
    children: PropTypes.node,
    content: PropTypes.bool,
    contentClass: PropTypes.string,
    darkTitle: PropTypes.bool,
    secondary: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object]),
    sx: PropTypes.object,
    title: PropTypes.oneOfType([PropTypes.node, PropTypes.string, PropTypes.object])
};

ColorSubCard.defaultProps = {
    content: true
};

export default ColorSubCard;
