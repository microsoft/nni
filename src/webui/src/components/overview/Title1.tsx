import * as React from 'react';
import {Stack} from 'office-ui-fabric-react';
import '../../static/style/overviewTitle.scss';
interface Title1Props {
    text: string;
    icon?: string;
    bgcolor?: string;
}

class Title1 extends React.Component<Title1Props, {}> {

    constructor(props: Title1Props) {
        super(props);
    }

    render(): React.ReactNode {
        const { text, icon, bgcolor } = this.props;
        return (
            <Stack horizontal className="panelTitle" style={{ backgroundColor: bgcolor }}>
                <img src={require(`../../static/img/icon/${icon}`)} alt="icon" />
                <span>{text}</span>
            </Stack>
        );
    }
}

export default Title1;