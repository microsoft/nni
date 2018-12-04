import * as React from 'react';
import { Link } from 'react-router';
import '../static/style/slideBar.scss';

class SlideBar extends React.Component<{}, {}> {

  render() {
    return (
      <ul className="nav">
        <li className="logo">
          <Link to={'/oview'}>
            <img src={require('../static/img/logo.png')} style={{ width: 156 }} alt="NNI logo" />
          </Link>
        </li>
        <li className="tab">
          <Link to={'/oview'} activeClassName="high">
            Overview
            </Link>
        </li>
        <li className="tab">
          <Link to={'/detail'} activeClassName="high">
            Trials Detail
            </Link>
        </li>
      </ul>
    );
  }
}

export default SlideBar;