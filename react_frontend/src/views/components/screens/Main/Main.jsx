import React from "react";
import "./style.css";
import {setModalQRCodeStatus} from 'store/actions';
 import { useSelector } from "react-redux";
import { useDispatch } from "react-redux";
import {
  Divider,
  InputAdornment,
  OutlinedInput,
  InputLabel,
  IconButton,
  Chip,
  Fab,
  Box,
  Grid,
} from "@mui/material";

export const Main = () => {
     const dispatch = useDispatch();
 
     const handleModalSearch  = () => {
         dispatch(setModalQRCodeStatus(true));
     }
     const handleKeyDown = (ev) => {
         console.log( {ev});
     }
  return (
    <div className="index">
      <div className="div">
        <div className="overlap">
          <div className="iphone-x-status-bars">
            <div className="iphone-x-status-bars-2">
              <div className="battery">
                <div className="overlap-group">
                  <div className="capacity" />
                </div>
                <img
                  className="cap"
                  alt="Cap"
                  src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c66e7be1d47795da781b6f/img/cap.svg"
                />
              </div>
              <img
                className="wifi"
                alt="Wifi"
                src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c66e7be1d47795da781b6f/img/wifi.svg"
              />
              <img
                className="cellular-connection"
                alt="Cellular connection"
                src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c66e7be1d47795da781b6f/img/cellular-connection.svg"
              />
              <div className="time-style">
                <p className="time">
                  <span className="text-wrapper">9:4</span>
                  <span className="span">1</span>
                </p>
              </div>
            </div>
          </div>
          <div className="overlap-2">
            <div className="overlap-3">
              <div className="group">
                <div className="text-wrapper-2">Portfolio Balance</div>
                <div className="element">$8,400.00</div>
              </div>
              <img
                className="line"
                alt="Line"
                src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c527ed8cf3a27c33b93919/img/line.svg"
              />
              <div className="group-2">
                <p className="p">Up by 4% from last month</p>
                <div className="overlap-group-wrapper">
                  <div className="overlap-group-2">
                    <div className="rectangle" />
                    <img
                      className="ion-ios-arrow-round"
                      alt="Ion ios arrow round"
                      src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c66e7be1d47795da781b6f/img/ion-ios-arrow-round-up.svg"
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className="overlap-wrapper">
              <div className="overlap-4">
                <div className="rectangle-2" />
                <img
                  className="ion-ios-add"
                  alt="Ion ios add"
                  src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c66e7be1d47795da781b6f/img/ion-ios-add.svg"
                />
              </div>
            </div>
          </div>
          <img
            className="feather-search"
            alt="Feather search"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c527ed8cf3a27c33b93919/img/feather-search.svg"
          />
          <Box onClick={handleModalSearch} onKeyDown={handleKeyDown} className="text-wrapper-4">
          <img
            className="screen-shot"
            alt="Screen shot"
            src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/screen-shot-2024-01-11-at-2-50-2.png"
          />
          </Box>
        </div>
        <div className="frame">
          <div className="overlap-group-3">
          <Box onClick={handleModalSearch} onKeyDown={handleKeyDown} className="text-wrapper-4">
            <img
              className="img"
              alt="Screen shot"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c527ed8cf3a27c33b93919/img/screen-shot-2024-01-08-at-4-46.png"
            />
            </Box>
          <Box onClick={handleModalSearch} onKeyDown={handleKeyDown} className="text-wrapper-4">
            <img
              className="screen-shot-2"
              alt="Screen shot"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c526a5d34cae0eb5f81cec/img/screen-shot-2024-01-08-at-4-46.png"
            />
            </Box>
          <Box onClick={handleModalSearch} onKeyDown={handleKeyDown} className="text-wrapper-4">
            <img
              className="screen-shot-3"
              alt="Screen shot"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c527ed8cf3a27c33b93919/img/screen-shot-2024-01-08-at-4-51.png"
            />
            </Box>
          <Box onClick={handleModalSearch} onKeyDown={handleKeyDown} className="text-wrapper-4">
            <img
              className="walgreens"
              alt="Walgreens"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c527ed8cf3a27c33b93919/img/walgreens-1.png"
            />
            </Box>
          <Box onClick={handleModalSearch} onKeyDown={handleKeyDown} className="text-wrapper-4">
            <img
              className="screen-shot-4"
              alt="Screen shot"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c527ed8cf3a27c33b93919/img/screen-shot-2024-01-08-at-4-51-1.png"
            />
            </Box>
            <div className="walgreens-2">
              <div className="div-wrapper">
                <div className="text-wrapper-3">Walgreens</div>
              </div>
              <div className="element-wrapper">
                <p className="element-2">
                  <span className="text-wrapper-4">
                    $1000
                    <br />
                  </span>
                  <span className="text-wrapper-5">7.15</span>
                </p>
              </div>
            </div>
            <div className="target">
              <div className="group-3">
                <div className="text-wrapper-6">Target</div>
              </div>
              <div className="element-wrapper">
                <p className="element-2">
                  <span className="text-wrapper-4">
                    $426
                    <br />
                  </span>
                  <span className="text-wrapper-7">3.15</span>
                </p>
              </div>
            </div>
            <div className="chipotle">
              <div className="group-4">
                <div className="text-wrapper-8">Chipotle</div>
              </div>
              <div className="element-wrapper">
                <p className="element-2">
                  <span className="text-wrapper-4">
                    $2212
                    <br />
                  </span>
                  <span className="text-wrapper-7">21.33</span>
                </p>
              </div>
            </div>
            <div className="homedepot">
              <div className="group-5">
                <div className="text-wrapper-9">Home Depot</div>
              </div>
              <div className="group-6">
                <p className="element-2">
                  <span className="text-wrapper-4">
                    $6612
                    <br />
                  </span>
                  <span className="text-wrapper-7">7.39</span>
                </p>
              </div>
            </div>
            <div className="starbucks">
              <div className="group-4">
                <div className="text-wrapper-8">Starbucks</div>
              </div>
              <div className="element-wrapper">
                <p className="element-2">
                  <span className="text-wrapper-4">
                    $1012
                    <br />
                  </span>
                  <span className="text-wrapper-7">11.33</span>
                </p>
              </div>
            </div>
            <img
              className="line-2"
              alt="Line"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c57b8248e55b500491bfa4/img/line-1.svg"
            />
            <img
              className="line-3"
              alt="Line"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c57b8248e55b500491bfa4/img/line-1.svg"
            />
            <img
              className="line-4"
              alt="Line"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c57b8248e55b500491bfa4/img/line-1.svg"
            />
            <img
              className="line-5"
              alt="Line"
              src="https://cdn.animaapp.com/projects/65c5267cd4b749ab51e73d7c/releases/65c57b8248e55b500491bfa4/img/line-1.svg"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

