import * as React from 'react';
import { Link, IndexLink } from 'react-router';
import { Icon } from 'antd';
import '../style/slideBar.css';

class SlideBar extends React.Component<{}, {}> {

  render() {
    return (
      <div className="slider">
        <ul className="nav">
          <li>
            <IndexLink to={'/oview'} activeClassName="high">
              <Icon className="icon" type="experiment" theme="outlined" />Overview
              <Icon className="floicon" type="right" />
            </IndexLink>
          </li>
          <li>
            <Link to={'/all'} activeClassName="high">
              <Icon className="icon" type="dot-chart" />Optimization Progress
              <Icon className="floicon" type="right" />
            </Link>
          </li>
          <li>
            <Link to={'/hyper'} activeClassName="high">
              <Icon className="icon" type="rocket" />Hyper Parameter
              <Icon className="floicon" type="right" />
            </Link>
          </li>
          <li>
            <Link to={'/trastaus'} activeClassName="high">
              <Icon className="icon" type="bar-chart" />Trial Status
              <Icon className="floicon" type="right" />
            </Link>
          </li>
          <li>
            <Link to={'/control'} activeClassName="high">
              <Icon className="icon" type="form" />Control
              <Icon className="floicon" type="right" />
            </Link>
          </li>
          <li>
            <a href="https://github.com/Microsoft/nni/issues" target="_blank">
              <Icon className="icon" type="smile" theme="outlined" />Feedback
            </a>
          </li>
        </ul>
      </div>
    );
  }
}

export default SlideBar;