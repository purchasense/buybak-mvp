import React from "react";
import { Link } from 'react-router-dom';
import "./style.css";
import { Box, Button } from '@mui/material';
import AnimateButton from 'ui-component/extended/AnimateButton';


export const QRCode = () => {
  return (
    <div className="index">
      <div className="div">
        <div className="overlap">
          <div className="rectangle" />
          <img
            className="group"
            alt="Group"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/group-2.png"
          />
          <div className="group-wrapper">
            <div className="group-2">
              <div className="overlap-group">
                <div className="div-wrapper">
                  <div className="group-3">
                    <div className="text-wrapper">Home Depot</div>
                  </div>
                </div>
              </div>
              <div className="group-4">
                <div className="element-sameer-wrapper">
                  <div className="element-sameer">
                    0x12345678&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                    Sameer_Kulkarni
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="iphone-x-status-bars">
            <div className="iphone-x-status-bars-2">
              <div className="battery">
                <div className="capacity-wrapper">
                  <div className="capacity" />
                </div>
                <img
                  className="cap"
                  alt="Cap"
                  src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/cap.svg"
                />
              </div>
              <img
                className="wifi"
                alt="Wifi"
                src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/wifi.svg"
              />
              <img
                className="cellular-connection"
                alt="Cellular connection"
                src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/cellular-connection.svg"
              />
              <div className="time-style">
                <p className="time">
                  <span className="span">9:4</span>
                  <span className="text-wrapper-2">1</span>
                </p>
              </div>
            </div>
          </div>
          <div className="element-wrapper">
            <p className="element">
              <span className="text-wrapper-3">
                $2492.00
                <br />
              </span>
              <span className="text-wrapper-4">&nbsp;&nbsp;&nbsp;&nbsp;7.85 shares</span>
            </p>
          </div>
          <img
            className="screen-shot"
            alt="Screen shot"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/screen-shot-2024-01-11-at-2-50-2.png"
          />
          <img
            className="img"
            alt="Screen shot"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/screen-shot-2024-01-08-at-4-46.png"
          />
        </div>
        <img
          className="group-5"
          alt="Group"
          src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/group-22.png"
        />
        <div className="overlap-2">
          <img
            className="group-6"
            alt="Group"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/group-13.png"
          />
        </div>
        <div className="screen-shot-wrapper">
		<Link to="/payed">
          <img
            className="screen-shot-2"
            alt="Screen shot"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/screen-shot-2024-02-02-at-2-15-1.png"
          />
		</Link>
        </div>
      </div>
    </div>
  );
};

