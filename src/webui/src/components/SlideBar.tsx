import * as React from 'react';
import { Link } from 'react-router';
import axios from 'axios';
import { MANAGER_IP } from '../static/const';
import MediaQuery from 'react-responsive';
import { DOWNLOAD_IP } from '../static/const';
import { Row, Col, Menu, Dropdown, Icon, Select, Button, Form } from 'antd';
import { FormComponentProps } from 'antd/lib/form';
import { OVERVIEWTABS, DETAILTABS, NNILOGO } from './stateless-component/NNItabs';
const { SubMenu } = Menu;
const { Option } = Select;
const FormItem = Form.Item;
import '../static/style/slideBar.scss';
import '../static/style/button.scss';

interface SliderState {
    version: string;
    menuVisible: boolean;
    navBarVisible: boolean;
    isdisabledFresh: boolean;
}

interface SliderProps extends FormComponentProps {
    changeInterval: (value: number) => void;
    changeFresh: (value: string) => void;
}

interface EventPer {
    key: string;
}

class SlideBar extends React.Component<SliderProps, SliderState> {

    public _isMounted = false;
    public divMenu: HTMLDivElement | null;
    public selectHTML: Select | null;

    constructor(props: SliderProps) {
        super(props);
        this.state = {
            version: '',
            menuVisible: false,
            navBarVisible: false,
            isdisabledFresh: false
        };
    }

    downExperimentContent = () => {
        axios
            .all([
                axios.get(`${MANAGER_IP}/experiment`),
                axios.get(`${MANAGER_IP}/trial-jobs`),
                axios.get(`${MANAGER_IP}/metric-data`)
            ])
            .then(axios.spread((res, res1, res2) => {
                if (res.status === 200 && res1.status === 200 && res2.status === 200) {
                    if (res.data.params.searchSpace) {
                        res.data.params.searchSpace = JSON.parse(res.data.params.searchSpace);
                    }
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    let trialMessagesArr = res1.data;
                    const interResultList = res2.data;
                    Object.keys(trialMessagesArr).map(item => {
                        // transform hyperparameters as object to show elegantly
                        trialMessagesArr[item].hyperParameters = JSON.parse(trialMessagesArr[item].hyperParameters);
                        const trialId = trialMessagesArr[item].id;
                        // add intermediate result message
                        trialMessagesArr[item].intermediate = [];
                        Object.keys(interResultList).map(key => {
                            const interId = interResultList[key].trialJobId;
                            if (trialId === interId) {
                                trialMessagesArr[item].intermediate.push(interResultList[key]);
                            }
                        });
                    });
                    const result = {
                        experimentParameters: res.data,
                        trialMessage: trialMessagesArr
                    };
                    const aTag = document.createElement('a');
                    const file = new Blob([JSON.stringify(result, null, 4)], { type: 'application/json' });
                    aTag.download = 'experiment.json';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'experiment.json';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            }));
    }

    downnnimanagerLog = () => {
        axios(`${DOWNLOAD_IP}/nnimanager.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const nniLogfile = res.data;
                    const aTag = document.createElement('a');
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    const file = new Blob([nniLogfile], { type: 'application/json' });
                    aTag.download = 'nnimanagerLog.log';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'nnimanagerLog.log';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            });
    }

    downDispatcherlog = () => {
        axios(`${DOWNLOAD_IP}/dispatcher.log`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200) {
                    const dispatchLogfile = res.data;
                    const aTag = document.createElement('a');
                    const isEdge = navigator.userAgent.indexOf('Edge') !== -1 ? true : false;
                    const file = new Blob([dispatchLogfile], { type: 'application/json' });
                    aTag.download = 'dispatcherLog.log';
                    aTag.href = URL.createObjectURL(file);
                    aTag.click();
                    if (!isEdge) {
                        URL.revokeObjectURL(aTag.href);
                    }
                    if (navigator.userAgent.indexOf('Firefox') > -1) {
                        const downTag = document.createElement('a');
                        downTag.addEventListener('click', function () {
                            downTag.download = 'dispatcherLog.log';
                            downTag.href = URL.createObjectURL(file);
                        });
                        let eventMouse = document.createEvent('MouseEvents');
                        eventMouse.initEvent('click', false, false);
                        downTag.dispatchEvent(eventMouse);
                    }
                }
            });
    }

    getNNIversion = () => {
        axios(`${MANAGER_IP}/version`, {
            method: 'GET'
        })
            .then(res => {
                if (res.status === 200 && this._isMounted) {
                    this.setState({ version: res.data });
                }
            });
    }

    handleMenuClick = (e: EventPer) => {
        if (this._isMounted) { this.setState({ menuVisible: false }); }
        switch (e.key) {
            // download experiment related content
            case '1':
                this.downExperimentContent();
                break;
            // download nnimanager log file
            case '2':
                this.downnnimanagerLog();
                break;
            // download dispatcher log file
            case '3':
                this.downDispatcherlog();
                break;
            case 'close':
            case '10':
            case '20':
            case '30':
            case '60':
                this.getInterval(e.key);
                break;
            default:
        }
    }

    handleVisibleChange = (flag: boolean) => {
        if (this._isMounted === true) {
            this.setState({ menuVisible: flag });
        }
    }

    getInterval = (value: string) => {

        if (value === 'close') {
            this.props.changeInterval(0);
        } else {
            this.props.changeInterval(parseInt(value, 10));
        }
    }

    menu = () => {
        return (
            <Menu onClick={this.handleMenuClick}>
                <Menu.Item key="1">Experiment Parameters</Menu.Item>
                <Menu.Item key="2">NNImanager Logfile</Menu.Item>
                <Menu.Item key="3">Dispatcher Logfile</Menu.Item>
            </Menu>
        );
    }

    // nav bar
    navigationBar = () => {
        const { version } = this.state;
        const feedBackLink = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Menu onClick={this.handleMenuClick} className="menu-list" style={{ width: 216 }}>
                {/* <Menu onClick={this.handleMenuClick} className="menu-list" style={{width: window.innerWidth}}> */}
                <Menu.Item key="feedback">
                    <a href={feedBackLink} target="_blank">Feedback</a>
                </Menu.Item>
                <Menu.Item key="version">Version: {version}</Menu.Item>
                <SubMenu
                    key="download"
                    onChange={this.handleVisibleChange}
                    title={
                        <span>
                            <span>Download</span>
                        </span>
                    }
                >
                    <Menu.Item key="1">Experiment Parameters</Menu.Item>
                    <Menu.Item key="2">NNImanager Logfile</Menu.Item>
                    <Menu.Item key="3">Dispatcher Logfile</Menu.Item>
                </SubMenu>
            </Menu>
        );
    }

    nniTabs = () => {
        return (
            <Menu className="menuModal">
                <Menu.Item key="overview"><Link to={'/oview'}>Overview</Link></Menu.Item>
                <Menu.Item key="detail"><Link to={'/detail'}>Trials detail</Link></Menu.Item>
            </Menu>
        );
    }

    refreshInterval = () => {
        const {
            form: { getFieldDecorator },
            // form: { getFieldDecorator, getFieldValue },
        } = this.props;
        return (
            <Form>
                <FormItem style={{ marginBottom: 0 }}>
                    {getFieldDecorator('interval', {
                        initialValue: 'Refresh every 10s',
                    })(
                        <Select onSelect={this.getInterval}>
                            <Option value="close">Disable Auto Refresh</Option>
                            <Option value="10">Refresh every 10s</Option>
                            <Option value="20">Refresh every 20s</Option>
                            <Option value="30">Refresh every 30s</Option>
                            <Option value="60">Refresh every 1min</Option>
                        </Select>,
                    )}
                </FormItem>
            </Form>
        );
    }

    select = () => {
        const { isdisabledFresh } = this.state;

        return (
            <div className="interval">
                <Button
                    className="fresh"
                    onClick={this.fresh}
                    type="ghost"
                    disabled={isdisabledFresh}
                >
                    <Icon type="sync" /><span>Refresh</span>
                </Button>
                {this.refreshInterval()}
            </div>
        );
    }

    fresh = (event: React.SyntheticEvent<EventTarget>) => {
        event.preventDefault();
        event.stopPropagation();
        if (this._isMounted) {
            this.setState({ isdisabledFresh: true }, () => {
                const whichPage = window.location.pathname;
                this.props.changeFresh(whichPage);
                setTimeout(() => { this.setState(() => ({ isdisabledFresh: false })); }, 1000);
            });
        }
    }

    desktopHTML = () => {
        const { version, menuVisible } = this.state;
        const feed = `https://github.com/Microsoft/nni/issues/new?labels=${version}`;
        return (
            <Row className="nav">
                <Col span={8}>
                    <span className="desktop-logo">{NNILOGO}</span>
                    <span className="left-right-margin">{OVERVIEWTABS}</span>
                    <span>{DETAILTABS}</span>
                </Col>
                <Col span={16} className="desktop-right">
                    <span>{this.select()}</span>
                    <span>
                        <Dropdown
                            className="dropdown"
                            overlay={this.menu()}
                            onVisibleChange={this.handleVisibleChange}
                            visible={menuVisible}
                            trigger={['click']}
                        >
                            <a className="ant-dropdown-link" href="#">
                                Download <Icon type="down" />
                            </a>
                        </Dropdown>
                    </span>
                    <span>
                        <a href={feed} target="_blank">
                            <img
                                src={require('../static/img/icon/issue.png')}
                                alt="NNI github issue"
                            />
                            Feedback
                        </a>
                    </span>
                    <span>
                        <span className="version">Version: {version}</span>
                    </span>
                </Col>
            </Row>
        );
    }

    tabeltHTML = () => {
        return (
            <Row className="nav">
                <Col className="tabelt-left" span={16}>
                    <span>
                        <Dropdown overlay={this.navigationBar()} trigger={['click']}>
                            <Icon type="unordered-list" className="more" />
                        </Dropdown>
                    </span>
                    <span className="left-right-margin tabelt-img">{NNILOGO}</span>
                    <span>{OVERVIEWTABS}</span>
                    <span className="left-margin">{DETAILTABS}</span>
                </Col>
                <Col className="tabelt-right" span={8}>
                    {this.select()}
                </Col>
            </Row>
        );
    }

    mobileHTML = () => {
        const { isdisabledFresh } = this.state;
        return (
            <Row className="nav">
                <Col className="left" span={8}>
                    <span>
                        <Dropdown className="more-mobile" overlay={this.navigationBar()} trigger={['click']}>
                            <Icon type="unordered-list" className="more" />
                        </Dropdown>
                    </span>
                    <span>
                        <Dropdown overlay={this.nniTabs()} trigger={['click']}>
                            <a className="ant-dropdown-link" href="#">
                                <span>NNI <Icon type="down" /></span>
                            </a>
                        </Dropdown>
                    </span>
                </Col>
                <Col className="center" span={8}>
                    <img
                        src={require('../static/img/logo2.png')}
                        alt="NNI logo"
                    />
                </Col>
                <Col className="right" span={8}>
                    <Button
                        className="fresh"
                        onClick={this.fresh}
                        type="ghost"
                        disabled={isdisabledFresh}
                    >
                        <Icon type="sync" /><span>Refresh</span>
                    </Button>
                </Col>
            </Row>
        );
    }

    componentDidMount() {
        this._isMounted = true;
        this.getNNIversion();
    }

    componentWillUnmount() {
        this._isMounted = false;
    }

    render() {
        const mobile = (<MediaQuery maxWidth={884}>{this.mobileHTML()}</MediaQuery>);
        const tablet = (
            <MediaQuery minWidth={885} maxWidth={1229}>{this.tabeltHTML()}</MediaQuery>
        );
        const desktop = (<MediaQuery minWidth={1230}>{this.desktopHTML()}</MediaQuery>);
        return (
            <div>
                {mobile}
                {tablet}
                {desktop}
            </div>
        );
    }
}

export default Form.create<FormComponentProps>()(SlideBar);