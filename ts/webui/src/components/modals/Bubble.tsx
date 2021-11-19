import * as React from 'react';
import PropTypes from 'prop-types';
import { DefaultButton } from '@fluentui/react/lib/Button';
import { TeachingBubble } from '@fluentui/react/lib/TeachingBubble';
import { DirectionalHint } from '@fluentui/react/lib/Callout';
import { useBoolean, useId } from '@uifabric/react-hooks';

export const TeachingBubbleRetiarii: React.FunctionComponent = (props: any) => {
  const {retiariiParam} = props;
  const buttonId = useId('targetButton');
  const [teachingBubbleVisible, { toggle: toggleTeachingBubbleVisible }] = useBoolean(false);

  return (
    <div>
      <DefaultButton
        id={buttonId}
        onClick={toggleTeachingBubbleVisible}
        text='Origin parameter'
      />

      {teachingBubbleVisible && (
        <TeachingBubble
          calloutProps={{ directionalHint: DirectionalHint.bottomCenter }}
          target={`#${buttonId}`}
          isWide={true}
          hasCloseButton={true}
          closeButtonAriaLabel="Close"
          onDismiss={toggleTeachingBubbleVisible}
          headline="Parameters"
        >
          {retiariiParam}
        </TeachingBubble>
      )}
    </div>
  );
};

TeachingBubbleRetiarii.propTypes = {
  retiariiParam: PropTypes.object
};
