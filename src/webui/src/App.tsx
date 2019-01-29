import * as React from 'react';
import { Row, Col } from 'antd';
import './App.css';
import SlideBar from './components/SlideBar';

class App extends React.Component<{}, {}> {
  render() {
    return (
      <Row className="nni" style={{minHeight: window.innerHeight}}>
        <Row className="header">
          <Col span={1} />
          <Col className="headerCon" span={22}>
            <SlideBar />
          </Col>
          <Col span={1} />
        </Row>
        <Row className="contentBox">
          <Row className="content">
            {this.props.children}
          </Row>
        </Row>
      </Row>
    );
  }
}

export default App;
